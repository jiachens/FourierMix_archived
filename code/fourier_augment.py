'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-30 16:37:09
LastEditors: Jiachen Sun
LastEditTime: 2022-01-30 17:44:41
'''
import torch
import fourier_basis
import numpy as np
import random
import torchvision
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
import time

pre = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip()
            ])

def generate_mask():
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.sqrt((i-15.5) ** 2 + (j-15.5) ** 2)**0.8)
    return mask
    
MASK = generate_mask()

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


def augment(x_orig, k=0, p=0, basis=None,chain = 3):
    # t = time.time()
    x_aug = torch.zeros_like(x_orig)
    mixing_weight_dist = Dirichlet(torch.empty(chain).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    for i in range(chain):
        # t = time.time()
        x_temp = augment_single(x_orig)
        # print('each aug', time.time() - t)
        x_aug += mixing_weights[i] * x_temp

    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_fourier = skip_conn_weight * x_aug + (1 - skip_conn_weight) * x_orig
    return x_fourier

def augment_single(x_orig):
    ######### Fourier #########
    #######NUMPY###########
    severity_1 = random.choice(range(1,6))
    severity_2 = random.choice(range(1,6))
    c = [0.2,0.3,0.4,0.5,0.6][severity_1-1] 
    d = [6,5,4,3,2][severity_2-1]
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
    ###################################

    ####### TORCH #######
    # x_orig = x_orig.cuda()
    # severity_1 = random.choice(range(1,6))
    # severity_2 = random.choice(range(1,6))
    # c = [0.2,0.3,0.4,0.5,0.6][severity_1-1]
    # d = [6,5,4,3,2][severity_2-1]
    # x_orig_1 = x_orig.clone()
    
    # t = time.time()
    # x_orig_f = torch.fft.fftn(x_orig_1, s=None, dim=(-2,-1), norm=None) 
    # x_orig_f_abs = torch.abs(x_orig_f) 
    # x_orig_f_ang = torch.angle(x_orig_f) 
    # flag = torch.sign(torch.rand() - 0.5)
    # x_orig_f_abs *= 1. + flag * torch.rand(*x_orig_f_abs.shape) * c * 3
    # # x_orig_f_abs += (np.random.uniform(*x_orig_f_abs.shape) - 0.5) * f * MASK 
    # x_orig_f_ang += (torch.rand(*x_orig_f_ang.shape) - 0.5) * np.pi / d * 3
    # x_orig_f.real = x_orig_f_abs * torch.cos(x_orig_f_ang)
    # x_orig_f.imag = x_orig_f_abs * torch.sin(x_orig_f_ang)
    # # row = np.random.choice(32,e,replace=True)
    # # col = np.random.choice(32,e,replace=True)
    # # x_orig_f[:,row,col] = 0
    # x_restored_1 = torch.abs(torch.fft.ifftn(x_orig_f, s=None, dim=(-2,-1), norm=None))
    # # x_restored_1 = torch.FloatTensor(x_restored_1) 
    # print('each aug',time.time() - t)
    #####################

    ######### Spatial #########
    severity_3 = random.choice(range(1,9))
    severity_4 = random.choice(range(1,9))
    severity_5 = random.choice(range(1,9))

    d = [0,0,0,0,5,10,15,20][severity_3-1]
    t = [None,None,None,None,(1/24.,1/24.),(1/12.,1/12.),(1/8.,1/8.),(1/6.,1/6.)][severity_4-1]
    s = [None,None,None,None,0.03,0.07,0.11,0.15][severity_5-1]
    
    space = torchvision.transforms.RandomAffine(degrees=d, translate=t, scale=None, shear=s)
    x_restored_2 = space(x_restored_1)
    ##############################

    # b = np.random.uniform()
    # # b = 1
    # x_restored = x_restored_1 * b + x_restored_2 * (1 - b)
    # # a = np.random.uniform()
    # # x_restored = x_restored * a + x_orig * (1-a)

    return x_restored_2