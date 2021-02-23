 # -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:56:34 2017

@author: Esra

Extracting features from pretrained VGG-16
The outputs result in 4096D vectors for each input image

"""

from warnings import catch_warnings
import numpy as np
from skimage.io import imread
from skimage.transform import rescale as resize
import matplotlib.pylab as plt
import tensorflow as tf
# this is from https://github.com/machrisaa/tensorflow-vgg 
import vgg16
import os.path
import pandas as pd
import tarfile
import glob
import os

def vgg_resizeimg(img):
    im_resized = resize(img, 224.0 / min(img.shape[0], img.shape[1]) + 0.0001)
    if im_resized.shape[0] > im_resized.shape[1]:
        diff = im_resized.shape[0] - 224
        im_resized = im_resized[int(diff / 2): 224 + int(diff / 2)]
    elif im_resized.shape[1] > im_resized.shape[0]:
        diff = im_resized.shape[1] - 224
        im_resized = im_resized[:, int(diff / 2): 224 + int(diff / 2)]
    else:
        pass
    return im_resized

# we have four images per location corresponding to different camera directions
# a = 0 degrees, b = 90 degrees, c = 180 degrees, d = 270 degrees
def convert_single_set(imname, vgg, sess):
    batch = []

    for k in ['a','b','c','d']:
        # imname = ../../data/tmp_images/<n>, k = "a", "b", "c", "d"
        imgk = imread('{}_{}.png'.format(imname, k))  # to get the images

        batchk = vgg_resizeimg(imgk).reshape((224, 224, 3))  # batchk.shape == (224, 224, 3)

        batch = batch + [batchk]  # to generate the input tensor
    batch = np.asarray(batch)  # (4, 224, 224, 3)

    """ vgg: instance of class Vgg16,
        vgg.fc6 will call the functions all way back to the function "self.build",
    """
    prob = sess.run(vgg.fc6, feed_dict={images: batch})  # get the 4096 nodes that fc6 outputs, shape == (4096,)
    return prob

nimgs = 133000  # this is the total number of images you would want to extract features from

missing_images = []  # container to store missing image

with tf.Session() as sess:
    images = tf.placeholder("float", [None, 224, 224, 3])

    # VGG network weights used: pretrained weights from https://github.com/machrisaa/tensorflow-vgg
    vgg = vgg16.Vgg16(vgg16_npy_path='../../models/pre-trained-networks/VGG16/vgg16.npy')
    
    vgg.build(images)  # build the well-trained model
    mod = 0

    for n in range(nimgs):
        if n % 1000 == 0:
            """ Our input images were stored in .tgz files for every 1000 image.
                We first untar(i.e. unzip) all pngs to a temporary image folder.(临时映像文件)
                This should be changed as necessary"""

            print("removing all png's from the tmp folder")

            # remove
            for f in glob.glob('../../data/tmp_images/*.png'):  # return a list of paths matching the pattern "... tmp_images/*.png"
                os.remove(f)  # remove the imaged that have been fed into the nn (see the details below)

            # copy
            file_name = '../../data/images/{}.tgz'.format(np.int(n / 1000))  # source files name
            print("extracting {}".format(file_name))

            tar = tarfile.open(file_name)  # ≈ open, open the source files
            tar.extractall('../../data/tmp_images/.')  # target folder
            tar.close()

        # get the 4096-D vector
        if not os.path.isfile('../../data/gview_codes/{}.npz'.format(n)):  # since the weights and the images are all stable, it's no need to pass through the nn twice
            # i.e. when there is not some file called "... gview_codes/n.npz", you come here
            if os.path.isfile('../../data/tmp_images/{}_a.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_b.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_c.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_d.png'.format(n)):
               # i.e. if those files are extracted successfully, you come here
                try:
                    # feed the images to the nn and get the 4096 nodes
                    prob = convert_single_set('../../data/tmp_images/{}'.format(n), vgg, sess)
                except:
                    print('conversion of {} did not work...'.format(n))
                    missing_images += [n]  # missing images while generating the 4096 nodes
                try:
                    np.savez_compressed('../../data/gview_codes/{}.npz'.format(n), code=prob)  # `prob` is saved as the name `code`, prob.shape == (4096,)
                    print("wrote '../../data/gview_codes/{}.npz'".format(n))
                except:
                    print('saving of {} did not work...'.format(n))
            else:
                missing_images += [n]  # missing image while extracting to the temporary image folder
        else:
            pass
        
        if n % 100 == 0:
            print('{}/{} done.'.format(n, nimgs))

"""
    # ! Files: ../../data/gview_codes/n.npz,
    where n = {0, ..., 133000},
    and every npz file contains a 4096-D vector, saved as the name 'code',
    representing the output of the conv-layers fed with the relative images
"""


