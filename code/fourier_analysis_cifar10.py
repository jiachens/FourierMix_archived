'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-15 18:55:35
LastEditors: Jiachen Sun
LastEditTime: 2021-09-14 21:30:33
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
#from fourier_augment import FourierDataset
from fourier_augment3 import FourierDataset
import torchvision
from torchvision import transforms




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

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_orig = get_dataset("cifar10", "train",scheme="fourier_half_ga")
    dataset_augmix = FourierDataset(dataset_orig,True)
    
    sum_ps2D = 0
    sum_ps2D_orig = 0
    sum_relative = 0
    sum_ps1D_corr = 0
    sum_ps1D_orig = 0

    orig = []
    corrupt = []
    res = []

    for i in range(100):
        (x, label) = dataset_augmix[i]
        (x_orig, label) = dataset_orig[i]

        orig.append(x_orig)
        corrupt.append(x)
        # x_orig = preprocess(x_orig)
        
        if x_orig.shape[0] != 32:
            x_orig = x_orig.permute(1,2,0)
        if x.shape[0] != 32:
            x = x.permute(1,2,0)

        x = x.numpy()
        x_orig = x_orig.numpy()


        
        # print(x - x_orig)
        img_grey = rgb2gray(np.round((x - x_orig) * 255) / 255)
        img_grey_corr = np.round((x) * 255) / 255
        img_grey_orig = np.round((x_orig) * 255) / 255

        img_grey_F_corr = np.fft.fftshift(np.fft.fft2(img_grey_corr))
        img_grey_F = np.fft.fftshift(np.fft.fft2(img_grey))
        img_grey_F_orig = np.fft.fftshift(np.fft.fft2(img_grey_orig))
        
        abs_corr = np.abs(img_grey_F_corr)
        ps2D = np.abs(img_grey_F)
        abs_orig = np.abs(img_grey_F_orig)

        # angle_corr = np.angle(img_grey_F_corr)
        # angle_orig = np.angle(img_grey_F_orig)

        # size = 8
        # start = (angle_corr.shape[0] - size) // 2
        # end = start + size
        # # print(abs_corr.shape)
        # abs_corr[start:end,start:end,:] = abs_orig[start:end,start:end,:]

        # img_grey_F_corr.real = abs_corr * np.cos(angle_corr)
        # img_grey_F_corr.imag = abs_corr * np.sin(angle_corr)
        # restored = np.abs(np.fft.ifft2(np.fft.ifftshift(img_grey_F_corr)))
        # restored = torch.FloatTensor(restored) 

        # res.append(restored.permute(2,0,1))

        # orig = torch.stack(orig)
        # corrupt = torch.stack(corrupt)
        # res = torch.stack(res)

        # test_img = torchvision.utils.make_grid(orig, nrow = 10)
        # torchvision.utils.save_image(
        #         test_img, "./test/test_0908/orig.png", nrow = 10
        #     )
        # test_img = torchvision.utils.make_grid(corrupt, nrow = 10)
        # torchvision.utils.save_image(
        #         test_img, "./test/test_0908/corrupt_" + args.corruption +  '_' + str(args.severity) + ".png", nrow = 10
        #     )

        # test_img = torchvision.utils.make_grid(res, nrow = 10)
        # torchvision.utils.save_image(
        #         test_img, "./test/test_0908/restore_" + args.corruption +  '_' + str(args.severity) + '_' + str(size) + ".png", nrow = 10
        #     )

        # ps1D_corr = azimuthalAverage(ps2D_corr)
        # ps1D_orig = azimuthalAverage(ps2D_orig)

        # sum_ps1D_corr += ps1D_corr
        # sum_ps1D_orig += ps1D_orig

        # relative = np.divide(ps2D,ps2D_orig)

        sum_ps2D += ps2D
        # sum_ps2D_orig += ps2D_orig
        # sum_relative += relative
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
        
    avg_ps2D = sum_ps2D / 100
    # avg_ps2D_orig = sum_ps2D_orig / len(dataset)
    # avg_relative = sum_relative / len(dataset)
    # avg_relative = np.divide(avg_ps2D,avg_ps2D_orig)
    # avg_ps1D_corr = sum_ps1D_corr / len(dataset)
    # avg_ps1D_orig = sum_ps1D_orig / len(dataset)

    avg_ps2D[16,16] = 0.
    # avg_relative[16,16] = 0.

    # plt.plot(np.divide(np.abs(avg_ps1D_corr-avg_ps1D_orig),avg_ps1D_orig), 'r')
    # plt.plot(avg_ps1D_orig, 'b')
    # plt.savefig('./test/new_fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_1d_relative_2.png',dpi=250,bbox_inches='tight')
    # ax = sns.heatmap(avg_relative,
    #             cmap="jet",
    #             cbar=True,
    #             vmin = 0., 
    #             vmax = 5.,
    #             # cbar_kws={"ticks":[]},
    #             xticklabels=False,
    #             yticklabels=False,)
    # plt.savefig('./test/new_fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_no_center_relative.png',dpi=250,bbox_inches='tight')

    ax = sns.heatmap(avg_ps2D,
                cmap="jet",
                cbar=True,
                vmin = 0., 
                vmax = 50.,
                # cbar_kws={"ticks":[]},
                xticklabels=False,
                yticklabels=False,)
    plt.savefig('./test/test_0908/fouriermix.png',dpi=250,bbox_inches='tight')
    # plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
    plt.close()

    corrupt = torch.stack(corrupt)
    test_img = torchvision.utils.make_grid(corrupt, nrow = 10)
    torchvision.utils.save_image(
            test_img, "./test/test_0908/corrupt_fourier_0912.png", nrow = 10
        )

    orig = torch.stack(orig)
    test_img = torchvision.utils.make_grid(orig, nrow = 10)
    torchvision.utils.save_image(
            test_img, "./test/test_0908/orig.png", nrow = 10
        )