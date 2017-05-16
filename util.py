import glob
import os

import dxchange
import tensorflow as tf
import numpy as np


def load_tiff_stack(path, rand_batch_size=None):

    filelist = glob.glob(os.path.join(path, '*.tif*'))
    if rand_batch_size is not None:
        filelist = np.random.choice(filelist, rand_batch_size).tolist().sort()
    temp = dxchange.read_tiff(filelist[0])
    shape = np.squeeze(temp).shape
    arr = np.zeros([len(filelist), shape[0], shape[1]])
    for (i, f) in enumerate(filelist):
        arr[i, :, :] = np.squeeze(dxchange.read_tiff(f))

    return arr


def load_tiff_data(x_path='data/train/x', y_path='data/train/y', rand_batch_size=None):

    x = load_tiff_stack(x_path, rand_batch_size)
    y = load_tiff_stack(y_path, rand_batch_size)

    return x, y