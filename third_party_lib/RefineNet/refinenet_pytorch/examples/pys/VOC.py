import sys
import os
import glob
import inspect
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

pardir = os.path.abspath(os.path.join(sys.argv[0], "../../../"))
sys.path.append(pardir)
from models.resnet import rf101
from utils.helpers import prepare_img


cmap = np.load(f'{pardir}/utils/cmap.npy')
img_paths = glob.glob(f'{pardir}/examples/img/*.jpg')
n_classes = 21

# Initialise models
model_inits = {
    'rf_101_voc': rf101,
}
models = dict()
for key, func in model_inits.items():
    net = func(n_classes, pretrained=True).eval()
    models[key] = net

# Figure 2 from the paper
n_cols = len(models) + 1  # 1 for image, 1 for groundtruth
n_rows = len(img_paths)

plt.figure(figsize=(16, 12))
idx = 1

with torch.no_grad():
    for img_path in img_paths:
        img = np.array(Image.open(img_path))
        # msk = cmap[np.array(Image.open(img_path.replace('jpg', 'png')))]
        orig_size = img.shape[:2][::-1]

        img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()

        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(img)
        plt.title('img')
        plt.axis('off')
        idx += 1

        # plt.subplot(n_rows, n_cols, idx)
        # plt.imshow(msk)
        # plt.title('gt')
        # plt.axis('off')
        # idx += 1

        for mname, mnet in models.items():
            segm = mnet(img_inp)[0].data.numpy().transpose(1, 2, 0)
            segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
            segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
            plt.subplot(n_rows, n_cols, idx)
            plt.imshow(segm)
            plt.title(mname)
            plt.axis('off')
            idx += 1
    plt.show()
