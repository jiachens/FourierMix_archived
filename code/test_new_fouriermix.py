'''
Description: 
Autor: Jiachen Sun
Date: 2021-09-08 22:05:50
LastEditors: Jiachen Sun
LastEditTime: 2021-09-09 11:56:58
'''
import numpy as np
import os
from PIL import Image
from numpy.lib.function_base import _angle_dispatcher
from skimage.color import rgb2gray
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
import datetime
from architectures import get_architecture
from torchvision.utils import save_image
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
from augment_and_mix import AugMixDataset, AutoDataset
import torchvision



parser = argparse.ArgumentParser(description='Fourier Analysis')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

if __name__ == "__main__":
    pass
    # if args.dataset == "cifar10-c":
    #     # print(args.path)
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    # elif args.dataset == "cifar10-c-bar":
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    
    dataset_orig = get_dataset("cifar10", "test")
    
    sum_ps2D = 0
    sum_ps2D_orig = 0
    sum_relative = 0
    sum_ps1D_corr = 0
    sum_ps1D_orig = 0

    orig = []
    corrupt = []
    res = []

    for i in range(100):
        # (x, label) = dataset[i]
        (x_orig, label) = dataset_orig[i]
        (x, label) = dataset_orig[i+2]

        orig.append(x_orig)
        # corrupt.append(x.permute(2,0,1))

        if x_orig.shape[0] != 32:
            x_orig = x_orig.permute(1,2,0)
        if x.shape[0] != 32:
            x = x.permute(1,2,0)

        x = x.numpy()
        x_orig = x_orig.numpy()


        
        # print(x - x_orig)
        # img_grey = np.round((x - x_orig) * 255) / 255
        img_grey_corr = np.round((x) * 255) / 255
        img_grey_orig = np.round((x_orig) * 255) / 255

        img_grey_F_corr = np.fft.fftshift(np.fft.fft2(img_grey_corr))
        # img_grey_F = np.fft.fftshift(np.fft.fft2(img_grey))
        img_grey_F_orig = np.fft.fftshift(np.fft.fft2(img_grey_orig))
        
        abs_corr = np.abs(img_grey_F_corr)
        # ps2D = np.abs(img_grey_F)
        abs_orig = np.abs(img_grey_F_orig)

        angle_corr = np.angle(img_grey_F_corr)
        angle_orig = np.angle(img_grey_F_orig)

        size = 32
        start = (angle_corr.shape[0] - size) // 2
        end = start + size
        # print(abs_corr.shape)
        abs_orig[start:end,start:end,:] = abs_corr[start:end,start:end,:] * 0.5 + abs_orig[start:end,start:end,:] * 0.5

        img_grey_F_orig.real = abs_orig * np.cos(angle_orig)
        img_grey_F_orig.imag = abs_orig * np.sin(angle_orig)
        restored = np.abs(np.fft.ifft2(np.fft.ifftshift(img_grey_F_orig)))
        restored = torch.FloatTensor(restored) 

        res.append(restored.permute(2,0,1))

    orig = torch.stack(orig)
    # corrupt = torch.stack(corrupt)
    res = torch.stack(res)

    test_img = torchvision.utils.make_grid(orig, nrow = 10)
    torchvision.utils.save_image(
            test_img, "./test/test_0908/orig_new.png", nrow = 10
        )
    # test_img = torchvision.utils.make_grid(corrupt, nrow = 10)
    # torchvision.utils.save_image(
    #         test_img, "./test/test_0908/corrupt_" + args.corruption +  '_' + str(args.severity) + ".png", nrow = 10
    #     )

    test_img = torchvision.utils.make_grid(res, nrow = 10)
    torchvision.utils.save_image(
            test_img, "./test/test_0908/restore_new.png", nrow = 10
        )