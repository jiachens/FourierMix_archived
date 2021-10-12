'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-07 15:15:28
LastEditors: Jiachen Sun
LastEditTime: 2021-10-12 12:25:18
'''
import random
import time
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torchvision import transforms
import numpy as np

import augmentations

transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor()
            ])

transform2=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ])

# transform3=transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandAugment(),
#                 transforms.ToTensor()
#             ])

# transform4=transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.TrivialAugmentWide(),
#                 transforms.ToTensor()
#             ])

class AutoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, no_jsd=False):
        self.dataset = dataset
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return transform(x), y
        else:
            return (transforms.RandomHorizontalFlip()(transforms.ToTensor()(x)), 
                    transform(x),
                    transform(x)), y

    def __len__(self):
        return len(self.dataset)

# class RandDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, no_jsd=False):
#         self.dataset = dataset
#         self.no_jsd = no_jsd

#     def __getitem__(self, i):
#         x, y = self.dataset[i]
#         if self.no_jsd:
#             return transform3(x), y
#         else:
#             return (transforms.RandomHorizontalFlip()(transforms.ToTensor()(x)), 
#                     transform3(x),
#                     transform3(x)), y

#     def __len__(self):
#         return len(self.dataset)

class GADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, no_jsd=False):
        self.dataset = dataset
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return transform2(transforms.ToTensor()(x)), y
        else:
            return (transform2(transforms.ToTensor()(x)), 
                    transform2(transforms.ToTensor()(x)),
                    transform2(transforms.ToTensor()(x))), y

    def __len__(self):
        return len(self.dataset)

# class GADataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, no_jsd=False):
#         self.dataset = dataset
#         self.no_jsd = no_jsd

#     def __getitem__(self, i):
#         x, y = self.dataset[i]
#         if self.no_jsd:
#             return transform(x), y
#         else:
#             return (transform2(transforms.ToTensor()(x)), 
#                     transform2(patchGaussian(x,transform)),
#                     transform2(patchGaussian(x,transform))), y

#     def __len__(self):
#         return len(self.dataset)

# def patchGaussian(x,preprocess):
#     # W = 25
#     x = preprocess(x)
#     row = np.random.randint(32)
#     col = np.random.randint(32)
#     start_row = max(0,row-13)
#     end_row = min(32,row+13)
#     start_col = max(0,col-13)
#     end_col = min(32,col+13)
#     x[:,start_row:end_row,start_col:end_col] += torch.randn_like(x[:,start_row:end_row,start_col:end_col]) * 0.25
#     x = torch.clamp(x,0.,1.)
#     return x


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation.
       referenced from https://github.com/google-research/augmix/blob/master/cifar.py
    """
    def __init__(self, dataset, preprocess, k, alpha, no_jsd=False, dataset_name=None):
        self.dataset = dataset
        self.preprocess = preprocess
        self.k = k
        self.alpha = alpha
        self.no_jsd = no_jsd
        self.dataset_name = dataset_name

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return augmentAndMix(x, self.k, self.alpha, self.preprocess, self.dataset_name), y
        else:
            return (self.preprocess(x), 
                    augmentAndMix(x, self.k, self.alpha, self.preprocess, self.dataset_name),
                    augmentAndMix(x, self.k, self.alpha, self.preprocess, self.dataset_name)), y

    def __len__(self):
        return len(self.dataset)

def augmentAndMix(x_orig, k, alpha, preprocess, dataset_name):
    # k : number of chains
    # alpha : sampling constant
    if dataset_name == "imagenet":
        augmentations.IMAGE_SIZE = 224 
    t = time.time()
    x_temp = x_orig # back up for skip connection

    x_aug = torch.zeros_like(preprocess(x_orig))
    mixing_weight_dist = Dirichlet(torch.empty(k).fill_(alpha))
    mixing_weights = mixing_weight_dist.sample()

    for i in range(k):
        sampled_augs = random.sample(augmentations.augmentations, k)
        aug_chain_length = random.choice(range(1,k+1))
        aug_chain = sampled_augs[:aug_chain_length]

        for aug in aug_chain:
            severity = random.choice(range(1,6))
            x_temp = aug(x_temp, severity)

        x_aug += mixing_weights[i] * preprocess(x_temp)

    skip_conn_weight_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * preprocess(x_orig)
    print('each aug', time.time() - t)
    return x_augmix


class ExpDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation.
       referenced from https://github.com/google-research/augmix/blob/master/cifar.py
    """
    def __init__(self, dataset, preprocess, exp, split, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.expert = exp
        self.no_jsd = no_jsd
        self.split = split

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.split == 'train':
            if self.no_jsd:
                return augment(x, self.expert, self.preprocess), y
            else:
                return (self.preprocess(x), 
                        augment(x, self.expert, self.preprocess),
                        augment(x, self.expert, self.preprocess)), y
        else:
            return augment(x, self.expert, self.preprocess), y

    def __len__(self):
        return len(self.dataset)


def augment(x_orig, exp, preprocess):
    # k : number of chains
    # alpha : sampling constant

    x_temp = x_orig # back up for skip connection

    x_aug = torch.zeros_like(preprocess(x_orig))

    # for i in range(k):
    #     sampled_augs = random.sample(augmentations, k)
    #     aug_chain_length = random.choice(range(1,k+1))
    #     aug_chain = sampled_augs[:aug_chain_length]
    if exp == 'autocontrast':
        aug = augmentations[0]
    elif exp == 'equalize':
        aug = augmentations[1]
    elif exp == 'posterize':
        aug = augmentations[2]
    elif exp == 'solarize':
        aug = augmentations[3]

    severity = random.choice(range(1,6))
    x_temp = aug(x_temp, severity)
    x_aug = preprocess(x_temp)
    #     x_aug += mixing_weights[i] * preprocess(x_temp)

    aug_chain_length = random.choice(range(1,6))
    aug_chain = augmentations.augmentations_x[:aug_chain_length]
    for aug in aug_chain:
        severity = random.choice(range(1,6))
        x_orig = aug(x_orig, severity)

    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * preprocess(x_orig)

    return x_augmix