'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-15 18:55:35
LastEditors: Jiachen Sun
LastEditTime: 2021-06-30 20:55:22
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
    
    sum_ps2D = 0
    sum_ps2D_orig = 0
    sum_relative = 0
    sum_ps1D_corr = 0
    sum_ps1D_orig = 0

    for i in range(len(dataset)):
        (x, label) = dataset[i]
        (x_orig, label) = dataset_orig[i]
        if x_orig.shape[0] != 32:
            x_orig = x_orig.permute(1,2,0)
        x = x.numpy()
        x_orig = x_orig.numpy()
        
        # print(x - x_orig)
        img_grey = rgb2gray(np.round((x - x_orig) * 255)) / 255
        img_grey_corr = rgb2gray(np.round((x) * 255)) / 255
        img_grey_orig = rgb2gray(np.round((x_orig) * 255)) / 255

        img_grey_F_corr = np.fft.fftshift(np.fft.fft2(img_grey_corr))
        img_grey_F = np.fft.fftshift(np.fft.fft2(img_grey))
        img_grey_F_orig = np.fft.fftshift(np.fft.fft2(img_grey_orig))
        
        ps2D_corr = np.abs(img_grey_F_corr)
        ps2D = np.abs(img_grey_F)
        ps2D_orig = np.abs(img_grey_F_orig)
        ps1D_corr = azimuthalAverage(ps2D_corr)
        ps1D_orig = azimuthalAverage(ps2D_orig)

        sum_ps1D_corr += ps1D_corr
        sum_ps1D_orig += ps1D_orig

        relative = np.divide(ps2D,ps2D_orig)

        sum_ps2D += ps2D
        sum_ps2D_orig += ps2D_orig
        sum_relative += relative
        # ax = sns.heatmap(ps2D_orig,
        #         cmap="jet",
        #         cbar=True,
        #         vmin = 0., 
        #         vmax = 40.,
        #         # cbar_kws={"ticks":[]},
        #         xticklabels=False,
        #         yticklabels=False,)
        # plt.savefig('./test/fourier_test/'+ str(i) +'.png',dpi=250,bbox_inches='tight')
        # plt.close()
        
    avg_ps2D = sum_ps2D / len(dataset)
    avg_ps2D_orig = sum_ps2D_orig / len(dataset)
    # avg_relative = sum_relative / len(dataset)
    avg_relative = np.divide(avg_ps2D,avg_ps2D_orig)
    avg_ps1D_corr = sum_ps1D_corr / len(dataset)
    avg_ps1D_orig = sum_ps1D_orig / len(dataset)

    # avg_ps2D[16,16] = 0.
    # avg_relative[16,16] = 0.

    plt.plot(np.divide(np.abs(avg_ps1D_corr-avg_ps1D_orig),avg_ps1D_orig), 'r')
    # plt.plot(avg_ps1D_orig, 'b')
    plt.savefig('./test/new_fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_1d_relative_2.png',dpi=250,bbox_inches='tight')
    # ax = sns.heatmap(avg_relative,
    #             cmap="jet",
    #             cbar=True,
    #             vmin = 0., 
    #             vmax = 5.,
    #             # cbar_kws={"ticks":[]},
    #             xticklabels=False,
    #             yticklabels=False,)
    # plt.savefig('./test/new_fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_no_center_relative.png',dpi=250,bbox_inches='tight')

    # ax = sns.heatmap(avg_ps2D,
    #             cmap="jet",
    #             cbar=True,
    #             vmin = 0., 
    #             vmax = 40.,
    #             # cbar_kws={"ticks":[]},
    #             xticklabels=False,
    #             yticklabels=False,)
    # plt.savefig('./test/fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_no_center_unified.png',dpi=250,bbox_inches='tight')
    # plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
    plt.close()