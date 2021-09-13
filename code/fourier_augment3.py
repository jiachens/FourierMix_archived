'''
Description: 
Autor: Jiachen Sun
Date: 2021-09-10 15:23:50
LastEditors: Jiachen Sun
LastEditTime: 2021-09-12 23:49:33
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

BAISIS = fourier_basis.generate_basis(1).cpu().numpy()


def generate_mask():
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.sqrt((i-15.5) ** 2 + (j-15.5) ** 2)**0.8)
    return mask
    
MASK = generate_mask()

def amplitude(x,y,sev):
    c = [0.2,0.3,0.4,0.5,0.6][sev-1]
    x_abs = np.abs(x)
    x_angle = np.angle(x)
    flag = np.sign(np.random.uniform(*x_abs.shape) - 0.5)
    x_abs *= 1. + flag * np.random.rand(*x_abs.shape) * c * MASK 
    x.real = x_abs * np.cos(x_angle)
    x.imag = x_abs * np.sin(x_angle)
    return x

def amplitude2(x,y,sev):
    c = [1.,1.25,1.5,1.75,2][sev-1] 
    x_abs = np.abs(x)
    x_angle = np.angle(x)
    # flag = np.random.uniform(*x_abs.shape) - 0.5)
    x_abs += (np.random.uniform(*x_abs.shape) - 0.5) * c * MASK 
    x.real = x_abs * np.cos(x_angle)
    x.imag = x_abs * np.sin(x_angle)
    return x

def phase(x,y,sev):
    c = [6,5,4,3,2][sev-1]
    x_ang = np.angle(x)
    x_abs = np.abs(x)
    x_ang += (np.random.rand(*x_ang.shape) - 0.5) * np.pi / c * MASK 
    x.real = x_abs * np.cos(x_ang)
    x.imag = x_abs * np.sin(x_ang)
    return x

def mixup(x,y,sev):
    c = [0.5,0.55,0.6,0.65,0.7][sev-1]
    y_abs = np.abs(y)
    x_abs = np.abs(x)
    x_ang = np.angle(x)

    x_abs = c * y_abs + (1-c) * x_abs
    x.real = x_abs * np.cos(x_ang)
    x.imag = x_abs * np.sin(x_ang)
    return x

def mask(x,y,sev):
    c = [20,40,60,80,100][sev-1]
    row = np.random.choice(32,c,replace=True)
    col = np.random.choice(32,c,replace=True)
    x[:,row,col] = 0
    return x

# def add(x,y,sev):
#     c = [10,20,30,40,50][sev-1]
#     row = np.random.choice(32,c,replace=True) 
#     col = np.random.choice(32,c,replace=True)
#     # x_restored = np.fft.ifft2(np.fft.ifftshift(x))
#     for i in range(c):
#         x[:,row[i],col[i]] *= 2
#     # x = np.fft.fft2(np.fft.fftshift(x_restored))
#     # x.imag = -x.imag
#     return x


# def mask(x,y,sev):
#     c = [28,25,22,19,16][sev-1]
#     start = (32 - c) // 2
#     end = start + c
#     x_temp = np.zeros_like(x)
#     x_temp[:,start:end,start:end] = x[:,start:end,start:end]
#     return x_temp
    


OP = [amplitude2,phase,mask]
# OP = [add,add,add,add]



class FourierDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, no_jsd=False):
        self.dataset = dataset
        self.no_jsd = no_jsd
        self.basis = None#fourier_basis.generate_basis(1).cpu()
    

    def __getitem__(self, i):
        x, y = self.dataset[i]
        idx = random.randint(0,len(self)-1)
        x_1 = self.dataset[idx][0]
        if self.no_jsd:
            return pre(augment(x,x_1)), y
        else:
            return (pre(x), 
                    pre(augment(x,x_1)),
                    pre(augment(x,x_1))), y

    def __len__(self):
        return len(self.dataset)


def augment(x_orig,x_1,chain = 3):

    x_aug = torch.zeros_like(x_orig)
    mixing_weight_dist = Dirichlet(torch.empty(chain).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    for i in range(chain):
        # chain2 = []
        # aug_chain_length = random.choice(range(1,5))
        # idxs = np.random.choice(4,aug_chain_length,replace=False)
        # for idx in idxs:
        #     chain2.append(OP[idx])
        chain2 = OP
        x_temp = augment_single(x_orig,x_1,chain2)
        x_aug += mixing_weights[i] * x_temp

    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_fourier = skip_conn_weight * x_aug + (1 - skip_conn_weight) * x_orig
    return x_fourier

def augment_single(x_orig,x_1,chain):

    ######### Spatial #########
    severity_3 = random.choice(range(1,9))
    severity_4 = random.choice(range(1,9))
    severity_5 = random.choice(range(1,9))
    severity_6 = random.choice(range(1,9))

    d = [0,0,0,0,5,10,15,20][severity_3-1]
    t = [None,None,None,None,(1/24.,1/24.),(1/12.,1/12.),(1/8.,1/8.),(1/6.,1/6.)][severity_4-1]
    s = [None,None,None,None,0.03,0.07,0.11,0.15][severity_5-1]
    s2 = [None,None,None,None,(0.975,1.025),(0.95,1.05),(0.925,1.075),(0.9,1.1)][severity_6-1]
    
    space = torchvision.transforms.RandomAffine(degrees=d, translate=t, scale=s2, shear=s)
    x_restored = space(x_orig)
    ##############################
    x_restored = x_restored.numpy()
    x_1 = x_1.numpy()


    x_f = np.fft.fftshift(np.fft.fft2(x_restored))
    x_1 = np.fft.fftshift(np.fft.fft2(x_1))

    for op in chain:
        severity = random.choice(range(1,6))
        x_f = op(x_f,x_1,severity)

    # # a = np.random.uniform()
    x_restored = np.fft.ifft2(np.fft.ifftshift(x_f))
    x_restored = torch.FloatTensor(x_restored)

    return x_restored