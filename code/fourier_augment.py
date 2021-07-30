'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-30 16:37:09
LastEditors: Jiachen Sun
LastEditTime: 2021-07-30 18:39:01
'''
import torch
import fourier_basis
import numpy as np


class FourierDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, k, p, no_jsd=False):
        self.dataset = dataset
        self.k = k
        self.p = p
        self.no_jsd = no_jsd
        self.basis = fourier_basis.generate_basis(1)
    

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return augment(x, self.k, self.p, self.basis), y
        else:
            return (x, 
                    augment(x, self.k, self.p, self.basis),
                    augment(x, self.k, self.p, self.basis)), y

    def __len__(self):
        return len(self.dataset)

def augment(x_orig, k, p, basis):

    for _ in range(p):
        row = np.random.choice(k) + ((31 - k) // 2)
        col = np.random.choice(k) + ((31 - k) // 2)
        perturbation = basis[:,2+row*34:(row+1)*34,2+col*34:(col+1)*34] * (2. / np.sqrt((row-15.5)**2+(col-15.5)**2)) * np.random.uniform(1., 2.)
        x_orig += perturbation * torch.tensor(np.random.choice((-1, 1),size=(3,1,1))).cuda()

    return x_orig