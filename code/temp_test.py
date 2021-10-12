'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-11 17:54:22
LastEditors: Jiachen Sun
LastEditTime: 2021-10-11 23:38:11
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

rootdir=os.path.join('../data/imagenet-sample-images')

preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomSizedCrop(224),
        transforms.ToTensor()
    ])

images = []

for (dirpath,dirnames,filenames) in os.walk(rootdir):
    for filename in filenames:
        if os.path.splitext(filename)[1]=='.JPEG':
            images.append(preprocess(Image.open(os.path.join(rootdir,filename))))

images = torch.stack(images[:63])
print(images.shape)
# images = torch.Tensor(np.random.randn(1000,3,224,224))
y = np.ones(63,)

train_dataset = [(images[i],y[i]) for i in range(63)]

train_data = FourierDataset(train_dataset, 0, 0, False)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32,
                              num_workers=4, pin_memory=True,sampler=None)

t = time.time()

for epoch in range(90):
    for i, (images, targets) in enumerate(train_loader):
        print(time.time() - t, i)
        t = time.time()
        print(images[1].shape)
        test_img = torchvision.utils.make_grid(images[1], nrow = 4)
        torchvision.utils.save_image(
            test_img, "../test/fourier/test_imagenet.png", nrow = 4
        )
