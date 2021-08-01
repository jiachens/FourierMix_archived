'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-29 20:52:26
LastEditors: Jiachen Sun
LastEditTime: 2021-08-01 15:52:44
'''
import numpy as np
import os
from PIL import Image
import argparse
from datasets import get_dataset, DATASETS, get_num_classes
import torchvision
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import fourier_basis
import random




parser = argparse.ArgumentParser(description='Fourier Analysis')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("outfile", type=str, help="output file")
# parser.add_argument("--path", type=str, help="path to dataset")
# parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
# parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
# parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


if __name__ == "__main__":
    # pass
    # if args.dataset == "cifar10-c":
    #     # print(args.path)
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    # elif args.dataset == "cifar10-c-bar":
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    
    dataset_orig = get_dataset("cifar10", "test")
    basis = fourier_basis.generate_basis(1)
    
    plot = []

    for i in range(100):
        (x_orig, label) = dataset_orig[i]
        # x_orig = x_orig.numpy()
        # x_orig += torch.randn_like(x_orig) * 0.25
        x_orig = x_orig.cuda()

        p = 50
        k = 15
        seen = set()
        x_orig_f = torch.fft.fftshift(torch.fft.fft2(x_orig))
        x_orig_f_2 = x_orig_f.clone()

        # for _ in range(p):
        #     r = np.random.uniform(0.,k) 
        #     theta = np.random.uniform(0.,2*np.pi) 
        #     row = int(r * np.cos(theta) + 15.5)
        #     col = int(r * np.sin(theta) + 15.5)
        #     if (row,col) in seen:
        #         continue
        #     else:
        #         seen.add((row,col))
        #     # perturbation = basis[:,2+row*34:(row+1)*34,2+col*34:(col+1)*34] * (1. / np.sqrt((row-15.5)**2+(col-15.5)**2)) #* np.random.uniform(1., 2.)
        #     # x_orig += perturbation * torch.tensor(np.random.choice((-1, 1),size=(3,1,1))).cuda()
        #     x_orig_f_2[:,row,col] *=  torch.tensor(np.random.uniform(0.25,0.5,size=(3,))).cuda()
        # x_restored_1 = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_orig_f_2)))
        # # print(np.random.choice((-1,1),size=(3,1,1)))
        # # x_restored += torch.randn_like(x_restored) * 0.25

        x_orig_f_abs = x_orig_f.abs() * (torch.randn_like(x_orig_f) + 1.)
        x_orig_f_ang = x_orig_f.angle() + torch.randn_like(x_orig_f) * np.pi / 5.
        # print(x_orig_f.real)
        x_orig_f.real = x_orig_f_abs * torch.cos(x_orig_f_ang)
        # print(x_orig_f.real)
        x_orig_f.imag = x_orig_f_abs * torch.sin(x_orig_f_ang)

        x_restored_2 = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_orig_f)))

        a = np.random.uniform()
        # x_restored = x_restored_1 * a + x_restored_2 * (1-a)
        plot.append(x_restored_2)
        
    plot = torch.stack(plot)
    test_img = torchvision.utils.make_grid(plot, nrow = 10)
    torchvision.utils.save_image(
        test_img, "./test/fourier/test_2.png", nrow = 10
    )
   
    plt.close()