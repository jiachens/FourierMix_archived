'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-19 21:42:07
LastEditors: Jiachen Sun
LastEditTime: 2021-10-19 22:22:57
'''
from fourier_augment import augment
import numpy as np
from torchvision.utils import save_image
import torchvision
import torch
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes

dataset_orig = get_dataset("cifar10", "test")

for i in range(len(dataset_orig)):
    (x_orig, label) = dataset_orig[i]
    plot = []
    for j in range(100):
        x_aug = augment(x_orig)
        plot.append(x_aug)
    plot = torch.stack(plot)
    test_img = torchvision.utils.make_grid(plot, nrow = 10)
    torchvision.utils.save_image(
        test_img, "./test/akshay/" + str(i) + ".png", nrow = 10
    )
    if i == 9:
        break
