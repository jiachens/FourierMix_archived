'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-20 16:54:37
LastEditors: Jiachen Sun
LastEditTime: 2021-07-29 22:43:35
'''
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2gray
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
import datetime
from architectures import get_architecture
from torchvision.utils import save_image
import torchvision
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns


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
    if args.dataset == "cifar10-c":
        # print(args.path)
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    elif args.dataset == "cifar10-c-bar":
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    
    dataset_orig = get_dataset("cifar10", "test")
    
    plot = []

    for i in range(100):
        (x, label) = dataset[i]
        (x_orig, label) = dataset_orig[i]
        if x.shape[0] != 3:
            x = x.permute(2,0,1)
        # x = x.numpy()
        x_orig = x
        # x_orig = x_orig.numpy()
        x_orig += torch.randn_like(x_orig) * 0.25
        # print(x - x_orig)
        # img_grey = rgb2gray(np.round((x - x_orig) * 255)) / 255
        # img_grey_corr = rgb2gray(np.round((x) * 255)) / 255
        # img_grey_orig = rgb2gray(np.round((x_orig) * 255)) / 255
        if True:
            x_orig_f = torch.fft.fftshift(torch.fft.fft2(x_orig))
        else:
            x_orig_f = torch.fft.fft2(x_orig)

        size = 31
        mask = torch.zeros_like(x_orig_f)
        start_id = (x_orig_f.shape[1] - size) // 2
        end_id = (x_orig_f.shape[1] + size) // 2

        mask[:,start_id:end_id,start_id:end_id] = 1.
        x_orig_f *= mask        

        if True:
            x_restored = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_orig_f)))
        else:
            x_restored = torch.abs(torch.fft.ifft2(x_orig_f))
            mean = torch.mean(x_restored)
            std = torch.std(x_restored)
            x_restored = 1./std * (x_restored - mean)

        plot.append(x_restored)
        
    plot = torch.stack(plot)
    test_img = torchvision.utils.make_grid(plot, nrow = 10)
    torchvision.utils.save_image(
        test_img, "./test/filter/test.png", nrow = 10
    )
   
    plt.close()