'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-29 20:52:26
LastEditors: Jiachen Sun
LastEditTime: 2021-07-30 20:37:58
'''
import numpy as np
import os
from PIL import Image
import argparse
from datasets import get_dataset, DATASETS, get_num_classes
import torchvision
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import fourier_basis
import random




parser = argparse.ArgumentParser(description='Fourier Analysis')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("outfile", type=str, help="output file")
# parser.add_argument("--path", type=str, help="path to dataset")
# parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
# parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
# parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


if __name__ == "__main__":
    # pass
    # if args.dataset == "cifar10-c":
    #     # print(args.path)
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    # elif args.dataset == "cifar10-c-bar":
    #     dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    
    dataset_orig = get_dataset("cifar10", "test")
    basis = fourier_basis.generate_basis(1)
    
    plot = []

    for i in range(100):
        (x_orig, label) = dataset_orig[i]
        # x_orig = x_orig.numpy()
        # x_orig += torch.randn_like(x_orig) * 0.25
        x_orig = x_orig.cuda()

        for _ in range(10):
            row = np.random.choice(10) + 10
            col = np.random.choice(10) + 10
            # print(basis.shape)
            perturbation = basis[:,2+row*34:(row+1)*34,2+col*34:(col+1)*34] *  (2. / np.sqrt((row-15.5)**2+(col-15.5)**2)) * np.random.uniform(1., 2.) 
            # print(perturbation.shape)
            # perturbation = perturbation
            x_orig += perturbation  * torch.tensor(np.random.choice((-1, 1),size=(3,1,1))).cuda()

        # x_orig += torch.randn_like(x_orig) * 0.25
        plot.append(x_orig)
        
    plot = torch.stack(plot)
    test_img = torchvision.utils.make_grid(plot, nrow = 10)
    torchvision.utils.save_image(
        test_img, "./test/fourier/test.png", nrow = 10
    )
   
    plt.close()