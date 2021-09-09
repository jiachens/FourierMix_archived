'''
Description: 
Autor: Jiachen Sun
Date: 2021-09-09 11:37:34
LastEditors: Jiachen Sun
LastEditTime: 2021-09-09 17:21:26
'''
import torch
import fourier_basis
import numpy as np
import random
import torchvision
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

pre = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip()
            ])


class FourierDataset2(torch.utils.data.Dataset):

    def __init__(self, dataset, m, no_jsd=False):
        self.dataset = dataset
        self.no_jsd = no_jsd
        self.m = m
        self.basis = None#fourier_basis.generate_basis(1).cpu()
    

    def __getitem__(self, i):
        x, y = self.dataset[i]
        idxs = np.random.choice(len(self),self.m,replace=False)
        pool = []
        for idx in idxs:
            pool.append(self.dataset[idx][0])

        if self.no_jsd:
            return pre(augment(x, self.basis, pool)), y
        else:
            return (pre(x), 
                    pre(augment(x, self.basis, pool)),
                    pre(augment(x, self.basis, pool))), y

    def __len__(self):
        return len(self.dataset)


def augment(x_orig,basis,pool,chain = 3):

    x_aug = torch.zeros_like(x_orig)
    mixing_weight_dist = Dirichlet(torch.empty(chain).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    for i in range(chain):
        x_temp = augment_single(x_orig,pool)
        x_aug += mixing_weights[i] * x_temp

    # skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    # skip_conn_weight = skip_conn_weight_dist.sample()

    # x_fourier = skip_conn_weight * x_aug + (1 - skip_conn_weight) * x_orig
    return x_aug

def augment_single(x_orig,pool):

    ######### Fourier #########
    mixing_weight_dist = Dirichlet(torch.empty(len(pool)).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample()
    x_orig = x_orig.clone().numpy()
    x_f = np.fft.fftshift(np.fft.fft2(x_orig))
    x_aug = np.zeros_like(x_orig)
    for i,img in enumerate(pool):
        img = img.clone().numpy()
        img_f = np.fft.fftshift(np.fft.fft2(img))
        img_f_abs = np.abs(img_f) 
        x_aug += float(mixing_weights[i]) * img_f_abs

    x_f.real = skip_conn_weight * x_f.real + (1-skip_conn_weight) * x_aug
    x_restored = np.abs(np.fft.ifft2(np.fft.ifftshift(x_f)))
    x_restored = torch.FloatTensor(x_restored) 

    return x_restored