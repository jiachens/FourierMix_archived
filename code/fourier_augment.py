'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-30 16:37:09
LastEditors: Jiachen Sun
LastEditTime: 2021-09-09 16:35:15
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


class FourierDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, k, p, no_jsd=False):
        self.dataset = dataset
        self.k = k
        self.p = p
        self.no_jsd = no_jsd
        self.basis = None#fourier_basis.generate_basis(1).cpu()
    

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return pre(augment(x, self.k, self.p, self.basis)), y
        else:
            return (pre(x), 
                    pre(augment(x, self.k, self.p, self.basis)),
                    pre(augment(x, self.k, self.p, self.basis))), y

    def __len__(self):
        return len(self.dataset)


def augment(x_orig, k, p, basis,chain = 3):

    x_aug = torch.zeros_like(x_orig)
    mixing_weight_dist = Dirichlet(torch.empty(chain).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    for i in range(chain):
        x_temp = augment_single(x_orig)
        x_aug += mixing_weights[i] * x_temp

    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_fourier = skip_conn_weight * x_aug + (1 - skip_conn_weight) * x_orig
    return x_fourier

def augment_single(x_orig):

    ######### Fourier #########
    severity_1 = random.choice(range(1,6))
    severity_2 = random.choice(range(1,6))
    # severity_3 = random.choice(range(1,6))
    c = [0.2,0.3,0.4,0.5,0.6][severity_1-1]
    d = [6,5,4,3,2][severity_2-1]
    #e = [22,20,18,16,14][severity_3-1]
    x_orig_1 = x_orig.clone().numpy()
    x_orig_f = np.fft.fftshift(np.fft.fft2(x_orig_1))
    x_orig_f_abs = np.abs(x_orig_f) 
    x_orig_f_ang = np.angle(x_orig_f) 
    flag = np.sign(np.random.uniform() - 0.5)
    x_orig_f_abs *= 1. + flag * np.random.rand(*x_orig_f_abs.shape) * c
    x_orig_f_ang += (np.random.rand(*x_orig_f_ang.shape) - 0.5) * np.pi / d
    x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
    x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
    x_restored_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))
    x_restored_1 = torch.FloatTensor(x_restored_1) 

    ######### Spatial #########
    severity_3 = random.choice(range(1,9))
    severity_4 = random.choice(range(1,9))
    severity_5 = random.choice(range(1,9))

    d = [0,0,0,0,5,10,15,20][severity_3-1]
    t = [None,None,None,None,(1/24.,1/24.),(1/12.,1/12.),(1/8.,1/8.),(1/6.,1/6.)][severity_4-1]
    s = [None,None,None,None,0.03,0.07,0.11,0.15][severity_5-1]
    
    space = torchvision.transforms.RandomAffine(degrees=d, translate=t, scale=None, shear=s)
    x_restored_2 = space(x_orig)
    ##############################

    b = np.random.uniform()
    # b = 0
    x_restored = x_restored_1 * b + x_restored_2 * (1 - b)

    # # a = np.random.uniform()
    # # x_restored = x_restored * a + x_orig * (1-a)

    return x_restored_1