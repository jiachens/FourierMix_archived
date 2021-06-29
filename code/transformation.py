'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-18 19:07:30
LastEditors: Jiachen Sun
LastEditTime: 2021-06-28 22:57:37
'''
from PIL import ImageEnhance
import numpy as np
import torch


class Contrast(object):

    def __init__(self, level, maxval):
        self.level = level 
        self.maxval = maxval #1.8

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



class Contrast_2(object):
    '''
    Contrast from https://github.com/bethgelab/imagecorruptions
    '''
    def __init__(self, severity):
        self.severity = severity 

    def __call__(self, img):

        return self.contrast(img)


    def contrast(self, img):
        c = [0.4, .3, .2, .1, .05][self.severity - 1]
        means = torch.mean(img, dim=(1, 2), keepdim=True)
        return torch.clamp((img - means) * c + means, 0., 1.)


    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity)


class Fog(object):
    def __init__(self,severity) -> None:
        self.severity = severity
        
    # modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
    def plasma_fractal(self,mapsize=256, wibbledecay=3):
        """
        Generate a heightmap using diamond-square algorithm.
        Return square 2d array, side length 'mapsize', of floats in range 0-255.
        'mapsize' must be a power of two.
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float_)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                        array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize,
                    stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize,
            0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()
    
    def next_power_of_2(self,x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def fog(self,x):
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][self.severity - 1]

        shape = x.shape
        max_side = torch.max(shape)
        map_size = self.next_power_of_2(int(max_side))

        max_val = torch.max(x)

        if len(shape) < 3 or shape[2] < 3:
            x += torch.Tensor(c[0] * self.plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                        :shape[0], :shape[1]])
        else:
            x += torch.Tensor(c[0] * \
                self.plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
                :shape[1]][..., np.newaxis])
        return torch.clamp(x * max_val / (max_val + c[0]), 0, 1) 