'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-30 16:37:09
LastEditors: Jiachen Sun
LastEditTime: 2021-08-17 14:39:23
'''
import torch
import fourier_basis
import numpy as np
import random
import torchvision

pre = torchvision.transforms.Compose([
                # torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip()
            ])


class FourierDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, k, p, no_jsd=False):
        self.dataset = dataset
        self.k = k
        self.p = p
        self.no_jsd = no_jsd
        self.basis = fourier_basis.generate_basis(1).cpu()
    

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

def augment(x_orig, k, p, basis):

    ######### Fourier #########
    p = np.random.choice(p)
    # k = np.random.choice(k) 
    severity_1 = random.choice(range(1,6))
    severity_2 = random.choice(range(1,6))
    c = [0.2,0.4,0.6,0.8,1.][severity_1-1]
    d = [8,7,6,5,4][severity_2-1]
    seen = set()

    x_orig_1 = x_orig.detach().numpy()
    x_orig_f = np.fft.fftshift(np.fft.fft2(x_orig_1))
    x_orig_f_abs = np.abs(x_orig_f) 
    x_orig_f_ang = np.angle(x_orig_f) 
    
    for _ in range(p):
        r = np.random.uniform(0.,k)
        theta = np.random.uniform(0.,2*np.pi)
        row = int(r * np.cos(theta) + 15.5)
        col = int(r * np.sin(theta) + 15.5)
        
        if (row,col) in seen:
            continue
        else:
            seen.add((row,col))
            
        x_orig_f_abs[:,row,col] *= 1. - np.random.rand(*x_orig_f_abs[:,row,col].shape) * c
        x_orig_f_ang[:,row,col] += (np.random.rand(*x_orig_f_abs[:,row,col].shape) - 0.5) * np.pi / d

    x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
    x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
    
    x_restored_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))
    x_restored_1 = torch.FloatTensor(x_restored_1)

    ######### Spatial #########
    severity_3 = random.choice(range(1,6))
    severity_4 = random.choice(range(1,6))
    severity_5 = random.choice(range(1,6))

    d = [0,5,10,15,20][severity_3-1]
    t = [None,(1/24.,1/24.),(1/12.,1/12.),(1/8.,1/8.),(1/6.,1/6.)][severity_4-1]
    s = [None,0.03,0.07,0.11,0.15][severity_5-1]
    
    space = torchvision.transforms.RandomAffine(degrees=d, translate=t, scale=None, shear=s)
    x_restored_2 = space(x_orig)
    ##############################

    b = np.random.uniform()
    x_restored = x_restored_1 * b + x_restored_2 * (1 - b)

    a = np.random.uniform()
    x_restored = x_restored * a + x_orig * (1-a)

    return x_restored