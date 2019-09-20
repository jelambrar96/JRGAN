# dataloader.py

import os

import numpy as np
# import scipy.misc
from PIL import Image

from glob import glob
# import matplotlib.pyplot as plt


class DataLoader:

    """
    """
    def __init__(self, dataset_name, img_res=(128, 128), downscale=0.25):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self._downscale = downscale
    """
    """
    def load_data(self, batch_size=1, is_testing=False):

        # data_type = "train" if not is_testing else "test"
        # save all images on list of paths
        dir_dataset = os.path.join(self.dataset_name, '*')
        print(dir_dataset)
        path = glob(dir_dataset)
        print(path)
        # select random
        batch_images = np.random.choice(path, size=batch_size)
        imgs_hr = []
        imgs_lr = []

        for img_path in batch_images:
            img = self.load_image(img_path)
            h, w = self.img_res
            low_h, low_w = int(h * self._downscale), int(w * self._downscale)

            img_hr = scipy.misc.imresize(img, self.img_res)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    """
    """
    def load_image(self, path):
        # return scipy.misc.imread(path, mode='RGB').astype(np.float)
        return np.array(Image.open(path), dtype=np.float)# .astype(np.float)

