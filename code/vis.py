'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-23 18:33:37
LastEditors: Jiachen Sun
LastEditTime: 2021-06-24 12:36:56
'''
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2gray
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
import datetime
from architectures import get_architecture
from torchvision.utils import save_image
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision


parser = argparse.ArgumentParser(description='Fourier Analysis')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if __name__ == "__main__":
    if args.dataset == "cifar10-c":
        # print(args.path)
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    elif args.dataset == "cifar10-c-bar":
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    
    dataset_orig = get_dataset("cifar10", "test")

    img = []
    img_orig = []

    for i in range(64):
        (x, label) = dataset[i]
        # save_image(
        #     x.permute(2,0,1), "./test/test.png"
        # )
        (x_orig, label) = dataset_orig[i]
        # save_image(
        #     x_orig, "./test/test_orig.png"
        # )
        img.append(x.permute(2,0,1).cuda())
        img_orig.append(x_orig.cuda())
    
    img = torch.stack(img)
    img_orig = torch.stack(img_orig)


    img = torchvision.utils.make_grid(img, nrow = 8)
    img_orig = torchvision.utils.make_grid(img_orig, nrow = 8)
    save_image(
        img, "./test/test_" + args.corruption + "_" + str(args.severity) + ".png", nrow = 8
    )
    save_image(
        img_orig, "./test/test_orig.png", nrow = 8
    )