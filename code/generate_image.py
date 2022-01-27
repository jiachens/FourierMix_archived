'''
Description: 
Autor: Jiachen Sun
Date: 2021-11-01 21:55:38
LastEditors: Jiachen Sun
LastEditTime: 2021-11-22 22:11:34
'''
import torch
import numpy as np
import random
import fourier_augment_cuda
from torchvision import transforms
import torchvision
from PIL import Image
import make_imagenet_c 
import os
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

# img = np.array(img.getdata())
# print(img.shape)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Scale(256),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip()
            ])


def augment(x_orig,device=None, k=0, p=0, basis=None,chain = 3):
    # t = time.time()
    # x_orig = x_orig.to(device)
    x_aug = torch.zeros_like(x_orig).to(device)
    mixing_weight_dist = Dirichlet(torch.empty(chain).fill_(1.))
    mixing_weights = mixing_weight_dist.sample()
    for i in range(chain):
        # t = time.time()
        x_temp = augment_single(x_orig,device=device)
        # print('each aug', time.time() - t)
        x_aug += mixing_weights[i] * x_temp

    skip_conn_weight_dist = Beta(torch.tensor([1.]), torch.tensor([1.]))
    skip_conn_weight = skip_conn_weight_dist.sample().cuda()
    # skip_conn_weight = 1
    x_fourier = skip_conn_weight * x_aug + (1 - skip_conn_weight) * x_orig
    return x_fourier

def augment_single(x_orig,device=None):
    ######### Fourier #########

    ####### TORCH #######
    # x_orig = x_orig.cuda()
    severity_1 = 5#random.choice(range(1,6))
    severity_2 = 5#random.choice(range(1,6))
    c = [0.3,0.4,0.5,0.7,0.8][severity_1-1]
    d = [6,5,4,3,2][severity_2-1]
    x_orig_1 = x_orig.detach().clone()
    # print(x_orig_1.shape)
    x_orig_f = torch.fft.fftn(x_orig_1, s=None, dim=(2,3), norm=None) 
    x_orig_f_abs = torch.abs(x_orig_f)
    # print(x_orig_f_abs)
    x_orig_f_ang = torch.angle(x_orig_f) 
    flag = np.sign(np.random.uniform() - 0.5)
    x_orig_f_abs *= 1. + flag * torch.rand(*x_orig_f_abs.shape).to(device) * c
    x_orig_f_ang += (torch.rand(*x_orig_f_ang.shape).to(device) - 0.5) * np.pi / d * 3
    # print(x_orig_f_ang)
    x_orig_f.real = x_orig_f_abs * torch.cos(x_orig_f_ang)
    x_orig_f.imag = x_orig_f_abs * torch.sin(x_orig_f_ang)
    # print(x_orig_f)
    x_restored_1 = torch.abs(torch.fft.ifftn(x_orig_f, s=None, dim=(2,3), norm=None))
    # print(x_restored_1 - x_orig_1)
    # print(x_orig_f-torch.fft.fftn(x_orig_1, s=None, dim=(-2,-1), norm=None))
    #####################
    # torchvision.utils.save_image(x_restored_1,'./test_1.png')

    ######### Spatial #########
    severity_3 = random.choice(range(1,9))
    severity_4 = random.choice(range(1,9))
    severity_5 = random.choice(range(1,9))

    d = [0,0,0,0,2.5,5,7.5,10][severity_3-1] 
    t = [None,None,None,None,(1/36.,1/36),(1/24.,1/24.),(1/12.,1/12.),(1/8.,1/8.)][severity_4-1] 
    s = [None,None,None,None,0.03,0.07,0.11,0.15][severity_5-1]
    
    space = torchvision.transforms.RandomAffine(degrees=d, translate=t, scale=None, shear=s)
    x_orig_2 = x_orig.detach().clone()
    x_restored_2 = space(x_orig_2)
    ##############################

    b = np.random.uniform()
    # b = 0.5
    x_restored = x_restored_1 * b + x_restored_2 * (1 - b)
    # torchvision.utils.save_image(x_restored,'./test.png')

    return x_restored

# img = make_imagenet_c.impulse_noise(np.transpose(img_tensor.numpy() * 255.,(1,2,0)),3)
# print(img.shape)
# img_tensor = torch.permute(img_tensor,(2, 0, 1)) / 255.

path = '/home/jiachens/AML4/smoothing/data/imagenet-sample-images'
i = 0
stack = []
for filename in os.listdir(path):
    img = Image.open(path + '/' + filename)
    

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,0)

    aug_img = augment(img_tensor.cuda(),device='cuda:0')
    aug_img = torch.squeeze(aug_img)
    if aug_img.shape[0] != 3:
        continue
    i += 1
    stack.append(aug_img)
    # print(aug_img.shape)
    if i == 100:
        break

stack = torch.stack(stack,dim=0)

print(stack.shape)

test_img = torchvision.utils.make_grid(stack, nrow = 10)
torchvision.utils.save_image(
        test_img,'./test.png', nrow = 10
    )