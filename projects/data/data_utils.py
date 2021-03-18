import math
import torch
import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from PIL import Image, ImageEnhance, ImageOps
from collections.abc import Sequence


class SubsetDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, indices, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.indices = indices
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = list(self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            # indices = list(range(len(self.dataset)))
            indices = self.indices


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# transformations
def handle_sample(sample):
    if not isinstance(sample, Sequence):
        sample = (sample,)
    return sample


def handle_size(size, img_size=None, keep_ratio=True):
    # both the returned `size` and `img_size` follow the convention (h, w)
    if isinstance(size, int):
        size = (size,)
    if len(size) == 1:
        if keep_ratio and img_size is not None:
            l = int(size / min(img_size) * max(img_size))
            size = (l, size[0]) if img_size[0] > img_size[1] else (size[0], l)
        elif not keep_ratio:
            size = (size[0], size[0])
    return size


def resize(img, size, interpolation, keep_ratio=True):
    img_size = list(img.size[::-1]) if isinstance(img, Image.Image) else list(img.shape[-2:])
    size = handle_size(size, img_size, keep_ratio)
    if isinstance(img, Image.Image):
        img = img.resize(size[::-1], interpolation)
    elif isinstance(img, torch.Tensor):
        raise "tensors are not currently supported."
    
    return img


def crop(img, upper, left, crop_height, crop_width):
    if isinstance(img, torch.Tensor):
        img = img[..., upper:upper+crop_height, left:left+crop_width]
    elif isinstance(img, Image.Image):
        img = img.crop((left, upper, left+crop_width, upper+crop_height))
    elif isinstance(img, np.ndarray):
        raise "imgs for ndarray are not currently supported"
    else:
        raise "img should be np.ndarray, torch.Tensor or Image.Image"
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for trans in self.transforms:
            if isinstance(trans, Transform):
                sample = trans(sample)
            elif isinstance(trans, Sequence):
                img = trans[0](sample[0])[0]
                tar = trans[1](sample[1])[0]
                sample = (img, tar)
            else:
                raise ( "Each transformation passed to `Compose` must be an isinstance of `Transform`,"
                        "or a 2-element sequence with each element applied on `img` and `target` respectively.")
        return sample


class Transform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        raise NotImplementedError


class DummyTransform(Transform):
    def __call__(self, sample):
        sample = handle_sample(sample)
        return sample


class PILToTensor(Transform):
    def __init__(self, map_to_01: bool = False):
        self.map_to_01 = map_to_01

    def __call__(self, sample):
        sample = handle_sample(sample)
        _sample = tuple()
        for item in sample:
            _item = np.array(item).reshape(item.size[1], item.size[0], len(item.getbands()))
            _item = torch.from_numpy(_item.transpose(2, 0, 1))
            if self.map_to_01:
                _item = _item.float().div(255.0)
            _sample += (_item,)
        return _sample


class RandomHorizontalFlip(Transform):
    def __call__(self, sample):
        sample = handle_sample(sample)
        _sample = tuple()
        if np.random.rand() < 0.5:
            for item in sample:
                _item = item.flip(-1) if isinstance(item, torch.Tensor) else item.transpose(Image.FLIP_LEFT_RIGHT)
                _sample += (_item,)
        else:
            _sample = sample
        return _sample


class Resize(Transform):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, sample):
        sample = handle_sample(sample)
        _sample = tuple()
        for item in sample:
            _item = resize(item, self.size, self.interpolation)
            _sample += (_item,)
        return _sample


class RandomSizeCrop(Transform):
    def __init__(self, out_size, area_scale=(0.1, 1.0), aspect_ratio_w2h=(0.75, 4.0/3.0), interpolation=Image.BILINEAR):
        self.out_size = handle_size(out_size, keep_ratio=False)
        self.interpolation = interpolation
        self.aspect_ratio_w2h = aspect_ratio_w2h
        self.area_scale = area_scale

    def get_params(self, img):
        img_height, img_width = list(img.size[::-1]) if isinstance(img, Image.Image) else list(img.shape[-2:])
        img_area = img_height * img_width
        img_scale = float(img_width)/float(img_height)
        self.area_scale = (1.0, 1.0) if self.area_scale is None else self.area_scale
        self.aspect_ratio_w2h = (img_scale, img_scale) if self.aspect_ratio_w2h is None else self.aspect_ratio_w2h

        for _ in range(10):
            rand_area = img_area * np.random.uniform(self.area_scale[0], self.area_scale[1])
            rand_ratio = np.exp(np.random.uniform(*np.log(self.aspect_ratio_w2h)))
            crop_width = int(round(np.sqrt(rand_area * rand_ratio)))
            crop_height = int(round(np.sqrt(rand_area / rand_ratio)))
            if 0 < crop_width <= img_width and 0 < crop_height <= img_height:
                upper = np.random.randint(0, img_height - crop_height + 1)
                left = np.random.randint(0, img_width - crop_width + 1)
                return upper, left, crop_height, crop_width

        img_ratio = float(img_width) / float(img_height)
        if img_ratio < min(self.aspect_ratio_w2h):
            crop_width = img_width
            crop_height = int(round(crop_width / min(self.aspect_ratio_w2h)))
        elif img_ratio > max(self.aspect_ratio_w2h):
            crop_height = img_height
            crop_width = int(round(crop_height * max(self.aspect_ratio_w2h)))
        else:
            crop_width = img_width
            crop_height = img_height
        upper = (img_height - crop_height) // 2
        left = (img_width - crop_width) // 2
        return upper, left, crop_height, crop_width

    def __call__(self, sample):
        sample = handle_sample(sample)
        upper, left, crop_height, crop_width = self.get_params(sample[0])
        _sample = tuple()
        for item in sample:
            _item = crop(item, upper, left, crop_height, crop_width)
            _item = resize(_item, self.out_size, self.interpolation)
            _sample += (_item,)
        return _sample


class Normalize(Transform):
    def __init__(self, mean=None, std=None):
        self.mean = [0., 0., 0.] if mean is None else mean
        self.std = [1., 1., 1.] if std is None else std
    
    def __call__(self, sample):
        sample = handle_sample(sample)
        _sample = tuple()
        for item in sample:
            if not isinstance(item, torch.Tensor):
                raise 'the input item should be a torch.Tensor'
            if item.ndim < 3:
                raise f'Expected item to be a tensor image of size (..., C, H, W). Got {item.size()}'

            if (item.dtype == torch.uint8):
                self.mean = [m*255.0 for m in self.mean]
                self.std = [s*255.0 for s in self.std]
            dtype, device, n_channel = item.dtype, item.device, item.shape[0]
            mean = torch.tensor(self.mean, dtype=dtype, device=device).view(n_channel, 1, 1)
            std = torch.tensor(self.std, dtype=dtype, device=device).view(n_channel, 1, 1)
            if (std == 0).any():
                raise "std evaluated to zero after converted to dtype: {}".format(item.dtype)
            _item = item.sub(mean).div(std)
            _sample += (_item,)
        return _sample


class CentralCrop(Transform):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, sample):
        sample = handle_sample(sample)
        img_size = list(sample[0].size[::-1]) if isinstance(sample[0], Image.Image) else list(sample[0].shape[-2:])
        crop_height, crop_width = handle_size(self.crop_size, img_size, True)
        upper = (img_size[0] - crop_height) // 2
        left = (img_size[1] - crop_width) // 2
        _sample = tuple()
        for item in sample:
            _item = crop(item, upper, left, crop_height, crop_width)
            _sample += (_item,)
        return _sample


def test():
    import glob
    import matplotlib.pyplot as plt

    files = glob.glob("./img/*.png")
    imgs = tuple(Image.open(f).convert("RGB") for f in files)
    transforms = Compose([
        RandomHorizontalFlip(),
        # Resize((512, 1024), Image.BILINEAR),
        # RandomSizeCrop([512, 1024], aspect_ratio_w2h=None),
        # CentralCrop((512, 1024)),
        (PILToTensor(map_to_01=True), PILToTensor()),
        (Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), DummyTransform()),
    ])
    imgs_ = transforms(imgs)
    for im in imgs_:
        plt.figure()
        plt.imshow(im.numpy().transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    test()

