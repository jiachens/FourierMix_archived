'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-11 17:54:22
LastEditors: Jiachen Sun
LastEditTime: 2021-10-11 18:16:16
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


images = torch.Tensor(np.random.randn(1000,3,224,224))
y = np.ones(1000,)
train_dataset = [(images[i],y[i]) for i in range(1000)]

train_data = FourierDataset(train_dataset, 0, 0, False)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32,
                              num_workers=8, pin_memory=True,sampler=None)

t = time.time()

for epoch in range(90):
    for i, (images, targets) in enumerate(train_loader):
        print(time.time() - t, i)
        t = time.time()
