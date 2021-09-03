'''
Description: 
Autor: Jiachen Sun
Date: 2021-09-01 21:19:03
LastEditors: Jiachen Sun
LastEditTime: 2021-09-03 01:23:49
'''
import random

import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torchvision import transforms

from make_cifar10_c import augmentation_c, d

convert_img = transforms.ToPILImage()
final_img = transforms.ToTensor()

class CorruptionDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, corruption, no_jsd=False):
        self.dataset = dataset
        self.preprocess = convert_img
        self.corruption = corruption
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return final_img(augment(x, self.corruption, self.preprocess)).float(), y
        else:
            return (final_img(augment(x, self.corruption, self.preprocess)).float(),
            final_img(augment(x, self.corruption, self.preprocess)).float(),
            final_img(augment(x, self.corruption, self.preprocess)).float()), y

    def __len__(self):
        return len(self.dataset)

def augment(x_orig, corruption, preprocess):

    corruption = d[corruption]
    severity = random.choice(range(1,6))
    x_aug = corruption(preprocess(x_orig),severity)

    return x_aug / 255.