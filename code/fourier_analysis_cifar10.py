'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-15 18:55:35
LastEditors: Jiachen Sun
LastEditTime: 2021-06-23 13:18:42
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
    for i in range(len(dataset)):
        (x, label) = dataset[i]
        # save_image(
        #     x.permute(2,0,1), "./test/test.png"
        # )
        (x_orig, label) = dataset_orig[i]
        # save_image(
        #     x_orig, "./test/test_orig.png"
        # )
        # x = x.cuda()
        if x_orig.shape[0] != 32:
            x_orig = x_orig.permute(1,2,0)
        x = x.numpy()
        x_orig = x_orig.numpy()
        
        # print(x - x_orig)
        img_grey = rgb2gray(np.round((x - x_orig) * 255)) / 255
        img_grey_orig = rgb2gray(np.round((x_orig) * 255)) / 255

        img_grey_F = np.fft.fftshift(np.fft.fft2(img_grey))
        img_grey_F_orig = np.fft.fftshift(np.fft.fft2(img_grey_orig))
        ps2D = np.abs(img_grey_F)
        ps2D_orig = np.abs(img_grey_F_orig)
        sum_ps2D += ps2D
        sum_ps2D_orig += ps2D_orig
        
    avg_ps2D = sum_ps2D / len(dataset)
    avg_ps2D_orig = sum_ps2D_orig / len(dataset)

    relative = np.divide(avg_ps2D,avg_ps2D_orig)
    # print(avg_ps2D.shape)
    avg_ps2D[16,16] = 0.
    # print('Max value: {}'.format(np.max(avg_ps2D)))
    # print('Min value: {}'.format(np.min(avg_ps2D)))
    # avg_ps2D = np.clip(avg_ps2D,0,2)

    ax = sns.heatmap(avg_ps2D,
                cmap="jet",
                cbar=True,
                vmin = 0., 
                vmax = 40.,
                # cbar_kws={"ticks":[]},
                xticklabels=False,
                yticklabels=False,)
    plt.savefig('./test/fourier_analysis/' + args.dataset + '_' + args.corruption +  '_' + str(args.severity) + '_no_center_unified.png',dpi=250,bbox_inches='tight')
    # plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
    plt.close()