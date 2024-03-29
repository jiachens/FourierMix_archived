'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-11 17:54:22
LastEditors: Jiachen Sun
LastEditTime: 2021-10-12 17:34:04
'''
import time
# import setGPU
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
import torchvision
import cifar10_c
import cifar10_c_bar
import cifar100_c
from architectures import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
import consistency
from fourier_augment import FourierDataset
from augment_and_mix import AugMixDataset, AutoDataset,GADataset
from PIL import Image
import random

# rootdir=os.path.join('../data/imagenet-sample-images')

# preprocess = transforms.Compose([
#         # transforms.ToPILImage(),
#         transforms.RandomSizedCrop(224),
#         transforms.ToTensor()
#     ])

# images = []

# for (dirpath,dirnames,filenames) in os.walk(rootdir):
#     for filename in filenames:
#         if os.path.splitext(filename)[1]=='.JPEG':
#             images.append(preprocess(Image.open(os.path.join(rootdir,filename))))

# images = torch.stack(images[:63])
# print(images.shape)
# # images = torch.Tensor(np.random.randn(1000,3,224,224))
# y = np.ones(63,)

# train_dataset = [(images[i],y[i]) for i in range(63)]

# train_data = FourierDataset(train_dataset, 0, 0, False)

# train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32,
#                               num_workers=4, pin_memory=True,sampler=None)

# t = time.time()

# for epoch in range(90):
#     for i, (images, targets) in enumerate(train_loader):
#         print(time.time() - t, i)
#         t = time.time()
#         print(images[1].shape)
#         test_img = torchvision.utils.make_grid(images[1], nrow = 4)
#         torchvision.utils.save_image(
#             test_img, "../test/fourier/test_imagenet.png", nrow = 4
#         )


x_orig_1 = np.random.randn(3,224,224)
t = time.time()
severity_1 = random.choice(range(1,6))
severity_2 = random.choice(range(1,6))
c = [0.2,0.3,0.4,0.5,0.6][severity_1-1]
d = [6,5,4,3,2][severity_2-1]
x_orig_f = np.fft.fftshift(np.fft.fft2(x_orig_1))
print(time.time() - t)
x_orig_f_abs = np.abs(x_orig_f) 
x_orig_f_ang = np.angle(x_orig_f) 
flag = np.sign(np.random.uniform() - 0.5)
print(time.time() - t)
x_orig_f_abs *= 1. + flag * np.random.rand(*x_orig_f_abs.shape) * c * 3
x_orig_f_ang += (np.random.rand(*x_orig_f_ang.shape) - 0.5) * np.pi / d * 3
print(time.time() - t)
x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
print(time.time() - t)
x_restored_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))
print(time.time() - t)



x_orig_1 = torch.Tensor(x_orig_1).cuda()
print(time.time() - t)
x_orig_f = torch.fft.fftn(x_orig_1, s=None, dim=(-2,-1), norm=None) 
x_orig_f_abs = torch.abs(x_orig_f) 
x_orig_f_ang = torch.angle(x_orig_f) 
x_orig_f_abs *= 1. + flag * torch.rand(*x_orig_f_abs.shape).cuda() * c * 3
x_orig_f_ang += (torch.rand(*x_orig_f_ang.shape).cuda() - 0.5) * np.pi / d * 3
x_orig_f.real = x_orig_f_abs * torch.cos(x_orig_f_ang)
x_orig_f.imag = x_orig_f_abs * torch.sin(x_orig_f_ang)
x_restored_1 = torch.abs(torch.fft.ifftn(x_orig_f, s=None, dim=(-2,-1), norm=None))
print(time.time() - t)