from __future__ import print_function, absolute_import, division
import os, sys, glob
from PIL import Image
from collections import namedtuple
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets.utils import extract_archive
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.preparation.json2labelImg import json2labelImg

from . import data_utils


class Cityscapes(Dataset):
    # Based on the PyTorch class `Cityscapes` in `~cityscapes.py`
    CityscapesClass = namedtuple(
        'CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color']
    )
    classes = [
        #                  name                     id    trainId  category          catId     hasInstances   ignoreInEval        color
        CityscapesClass(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        CityscapesClass(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        CityscapesClass(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        CityscapesClass(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        CityscapesClass(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        CityscapesClass(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        CityscapesClass(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        CityscapesClass(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        CityscapesClass(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        CityscapesClass(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        CityscapesClass(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        CityscapesClass(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        CityscapesClass(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        CityscapesClass(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        CityscapesClass(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        CityscapesClass(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        CityscapesClass(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        CityscapesClass(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        CityscapesClass(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        CityscapesClass(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        CityscapesClass(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        CityscapesClass(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        CityscapesClass(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        CityscapesClass(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        CityscapesClass(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        CityscapesClass(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        CityscapesClass(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        CityscapesClass(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        CityscapesClass(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        CityscapesClass(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        CityscapesClass(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        CityscapesClass(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        CityscapesClass(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        CityscapesClass(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        CityscapesClass(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    def __init__(self, root, split="train", ds_type="gtFine", transforms=None, convert_id=False):
        assert ds_type in ("gtFine", "gtCoarse")
        valid_splits = ("train", "test", "val") if ds_type == "gtFine" else ("train", "train_extra", "val")
        assert split in valid_splits

        self.images = []
        self.targets = []
        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transforms = transforms
        self.images_path = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_path = os.path.join(self.root, self.ds_type, split)

        if convert_id:
            self.convert_to_train_id()

        for city in os.listdir(self.images_path)[:1]:
            img_path = os.path.join(self.images_path, city)
            target_path = os.path.join(self.targets_path, city)
            for img_name in os.listdir(img_path)[:20]:
                target_name = img_name.split('_leftImg8bit')[0] + f'_{self.ds_type}_labelTrainIds.png'
                self.images.append(os.path.join(img_path, img_name))
                self.targets.append(os.path.join(target_path, target_name))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])
        if self.transforms is not None:
            image, target = self.transforms((image, target))
        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Data Type: {ds_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def convert_to_train_id(self):
        # Based on https://github.com/mcordts/cityscapesScripts
        fnames_fine = os.path.join(self.root , "gtFine"   , "*" , "*" , "*_gt*_polygons.json")
        fnames_coarse = os.path.join(self.root , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json")
        files_fine = glob.glob(fnames_fine)
        files_coarse = glob.glob(fnames_coarse)
        files_fine.sort()
        files_coarse.sort()
        files = files_fine + files_coarse
        if not files:
            raise (f"Did not find any files in {self.root}")

        print("Processing {} annotation files".format(len(files)))
        for i, json_file in enumerate(files):
            img_file = json_file.replace("_polygons.json" , "_labelTrainIds.png")
            if not os.path.isfile(img_file):
                try:
                    json2labelImg(json_file, img_file, "trainIds")
                except:
                    print("[WARNING]: Failed to convert {}".format(json_file))
                    continue
                print("\rProgress: {:.2f} %".format(i * 100 / len(files)), end=' ')
            sys.stdout.flush()


def extract_zip_file(root, use_train_extra):
    print("\n==> Extracting data from zip file...", end=" ", flush=True)
    suffix = ['leftImg8bit_trainvaltest.zip', 'leftImg8bit_trainextra.zip', 'gtFine_trainvaltest.zip', 'gtCoarse.zip']
    for suf in suffix:
        from_path = os.path.join(root, suf)
        if os.path.isfile(from_path):
            extract_archive(from_path=from_path, to_path=root)
        else:
            raise RuntimeError(
                'Dataset not found or incomplete. Please make sure all required folders'
                ' for the specified "split" and "ds_type" are inside the "root" directory'
            )
    print("Extraction finished!", flush=True)


def load_data(args):
    print("\n==> Loading data ...", end=" ", flush=True)
    if args.extract_zip:
        extract_zip_file(args.data_path, args.use_train_extra)
    pin_memory = False if args.gpus is None else True
    Normalize = data_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    transform_train = data_utils.Compose([
        data_utils.RandomHorizontalFlip(),
        data_utils.RandomSizeCrop((512, 1024), aspect_ratio_w2h=None),
        (data_utils.PILToTensor(True), data_utils.PILToTensor()),
        # (Normalize, data_utils.DummyTransform()),
    ])
    trainset = Cityscapes(args.data_path, "train", "gtFine", transforms=transform_train)
    if args.use_train_extra:
        coarseset = Cityscapes(args.data_path, "train_extra", "gtCoarse", transforms=transform_train)
        trainset = ConcatDataset([trainset, coarseset])

    train_loader = DataLoader(
        trainset, args.batch_size_train, shuffle=True,
        num_workers=2, pin_memory=pin_memory,
    )

    transform_test = data_utils.Compose([
        data_utils.Resize((512, 1024)),
        (data_utils.PILToTensor(True), data_utils.PILToTensor()),
        # (Normalize, data_utils.DummyTransform()),
    ])
    testset = Cityscapes(args.data_path, "val", "gtFine", transforms=transform_test)
    test_loader = DataLoader(
        testset, args.batch_size_train, shuffle=False,
        num_workers=2, pin_memory=pin_memory,
    )
    print("Data loading finished!", flush=True)
    return train_loader, test_loader


def test():
    sys.path.append("../")
    import matplotlib.pyplot as plt
    from utils.options import args

    args.data_path = "F:/Self-Development/Academic/Computer_Science/Datasets/torch/torchvision/Cityscapes"
    args.gpus = None
    args.ds_type = "gtFine"
    train_loader, test_loader = load_data(args)
    sys.exit()
    for img, label in train_loader:
        img, label = img[0].numpy().transpose(1, 2, 0), label[0].numpy().transpose(1, 2, 0)
        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.imshow(label)
        plt.show()
        break


if __name__ == "__main__":
    test()
