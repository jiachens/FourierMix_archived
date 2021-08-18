'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-29 22:44:13
LastEditors: Jiachen Sun
LastEditTime: 2021-08-18 03:17:22
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
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--severity", type=int, default=1, help="severity level")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


if __name__ == "__main__":
    
    dataset_orig = get_dataset("cifar10", "test")
    
    plot = []
    labels = []

    for i in range(len(dataset_orig)):

        (x_orig, label) = dataset_orig[i]
        x_orig = x_orig.detach().numpy()

        x_orig_f = np.fft.fftshift(np.fft.fft2(x_orig))
        x_orig_f_abs = np.abs(x_orig_f)
        x_orig_f_ang = np.angle(x_orig_f)
        x_orig_f_abs *=  1 - np.random.rand(*x_orig_f_abs.shape) * 0.4
        # print(x_orig_f_abs/np.abs(x_orig_f))
        # x_orig_f_ang += (np.random.rand(*x_orig_f_abs.shape) - 0.5) * np.pi / 2

        x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
        x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
        
        x_restored = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))

        plot.append(x_restored)
        labels.append(label)
        
    plot = np.clip((np.stack(plot) * 255).astype(int), 0,255)
    # print(plot)

    plot = np.transpose(plot, (0, 2, 3, 1))
    labels = np.array(labels)
    os.makedirs('./data/CIFAR-10-F',exist_ok = True)
    
    np.save('./data/CIFAR-10-F/test.npy',plot)
    np.save('./data/CIFAR-10-F/label.npy',labels)

    # plot = torch.FloatTensor(plot).permute(0,3,1,2)
    
    # test_img = torchvision.utils.make_grid(plot[:100], nrow = 10)
    # torchvision.utils.save_image(
    #     test_img, "./test/filter/test_fft.png", nrow = 10
    # )
   
    # plt.close()