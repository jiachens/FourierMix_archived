'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-18 19:07:30
LastEditors: Jiachen Sun
LastEditTime: 2021-06-18 19:44:14
'''
from PIL import ImageEnhance
import numpy as np


class Contrast(object):

    def __init__(self, level, maxval):
        self.level = level
        self.maxval = maxval

    def __call__(self, pil_img):

        return self.contrast(pil_img)


    def contrast(self, pil_img):
        '''
        https://github.com/google-research/augmix/blob/master/augmentations.py
        '''
        level = self.float_parameter(self.sample_level(self.level)) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    def float_parameter(self, level):
        """Helper function to scale `val` between 0 and maxval.
        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(level) * self.maxval / 10.

    def sample_level(self, n):
        return np.random.uniform(low=0.1, high=n)


    def __repr__(self):
        return self.__class__.__name__ + '(level={0}, maxval={1})'.format(self.level, self.maxval)


