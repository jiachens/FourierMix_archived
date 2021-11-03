'''
Description: 
Autor: Jiachen Sun
Date: 2021-11-01 21:55:38
LastEditors: Jiachen Sun
LastEditTime: 2021-11-01 23:04:58
'''
import torch
import numpy as np
import random
import fourier_augment_cuda
from torchvision import transforms
import torchvision
from PIL import Image
import make_imagenet_c 

img = Image.open("./analysis/n02113186_19779.JPEG")

# img = np.array(img.getdata())

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomSizedCrop(224),
                # transforms.RandomHorizontalFlip()
            ])

img_tensor = transform(img)
# img_tensor = torch.unsqueeze(img_tensor,0)

img = make_imagenet_c.impulse_noise(np.transpose(img_tensor.numpy() * 255.,(1,2,0)),3)
print(img.shape)
img_tensor = torch.permute(torch.Tensor(img),(2, 0, 1)) / 255.

# aug_img = fourier_augment_cuda.augment(img_tensor.cuda(),device='cuda:0')

# print(aug_img.shape)

torchvision.utils.save_image(img_tensor,'./test.png')