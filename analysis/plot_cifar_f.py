'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-10 21:59:29
LastEditors: Jiachen Sun
LastEditTime: 2021-10-10 22:50:27
'''
import os
import numpy as np
import torchvision
import torch

_DIRNAME = 'CIFAR-10-F'
_LABELS_FILENAME = 'label.npy'

def generate_examples(data_dir='../data', corruption=None, severity=2):
    severity = severity # (1,2,3,4,5)
    data_dir = os.path.join(data_dir,_DIRNAME)
    images_file = os.path.join(data_dir, 'fourier_' + corruption + '.npy')
    labels_file = os.path.join(data_dir, _LABELS_FILENAME)
    labels = np.load(labels_file)
    num_images = labels.shape[0] // 3
    labels = labels[:num_images]
    images = np.load(images_file)
    images = images[(severity - 1) * num_images:severity * num_images]
    return images[:200] / 255.

imag = []
for alpha in [0.1, 0.5, 1, 2, 3]:
    for i in range(1,17):
        images = generate_examples(corruption=str(i)+"_"+str(alpha))
        imag.append(images)

imag = np.stack(imag)
imag = torch.Tensor(imag)

for j in range(100):
            
    test_img = torchvision.utils.make_grid(imag[:,j,:,:,:], nrow = 16)
    torchvision.utils.save_image(
            test_img, "../test/cifar10-f/" + str(j) + ".png", nrow = 16
        )