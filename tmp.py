import sys, os
import cv2
import torch
import torchvision
import numpy as np
import requests

from urllib.request import urlopen
import matplotlib.pyplot as plt

# model = resnet.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))

def test():

    def download(url, name):
        conn = urlopen(url)
        outimg = conn.read()
        conn.close()
        data_img = cv2.imdecode(np.asarray(bytearray(outimg), dtype=np.uint8), 1)
        if data_img is not None:
            if not os.path.isfile(name):
                cv2.imwrite(name, data_img)
                plt.imshow(data_img)    
                return data_img
    save_dir = "./out_image/beijing_new/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open("./tmp.txt","r") as f:
        start_end_points = f.readlines()
    start_point, end_point = start_end_points
    start_point_jin, start_point_wei = start_point.strip('\n').split(',')
    end_point_jin, end_point_wei = end_point.strip('\n').split(',')

    jins = np.arange(float(start_point_jin)*10000, float(end_point_jin)*10000, 1)*0.0001
    weis = np.linspace(float(start_point_wei)*10000, float(end_point_wei)*10000, len(jins))*0.0001

    print("==> processing {} requests...".format(len(jins)))
    for i, (jin, wei) in enumerate(zip(jins, weis)):
        img_name = "./out_image/beijing_new/" + str(round(jin, 6)) + "_" + str(round(wei, 6)) +".jpg"
        url = "http://api.map.baidu.com/panorama/v2?ak=9FcRfTXGEkpiBMrkjV7d2BGOVBXcaAoo&width=1024&height=512&location="+str(jin)+","+str(wei)+"&fov=180"
        outimg = download(url, img_name)


# help(torchvision.datasets.utils.extract_archive)

def mul_mod(a, b, n):
    return a * b - n * (a * b // n)

def check_closed(set_, op, mod=None):
    for i in set_:
        for j in set_:
            result = op(i, j, mod)
            if result not in set_:
                print(f"{i} {j}={result} is not in {set_}")


# check_closed([1, 9, 16, 22, 53, 74, 79, 81], mul_mod, 91)


print(torch.unique(torch.tensor([1, 2, 255, 4, 2])))


