# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""

import os
import numpy as np
from PIL import Image
from natsort import natsorted


def convert_one_channel(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img

def optimized_pre_images(resize_shape, path):

    # List all file in path
    dirs = natsorted(os.listdir(path))

    number_of_images = len(dirs)
    # Save size of images
    sizes = np.zeros((len(dirs), 2))

    images = img = Image.open(os.path.join(path, dirs[0]))
    sizes[0, :] = images.size
    images = images.resize((resize_shape), Image.Resampling.LANCZOS)
    images = convert_one_channel(np.asarray(images))
    
    for i in range(1, number_of_images):
        img = Image.open(os.path.join(path, dirs[i]))
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.Resampling.LANCZOS)
        img = convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))

    images = np.reshape(images, (number_of_images, *resize_shape, 1))
    return images, sizes

