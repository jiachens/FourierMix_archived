'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-29 22:44:13
LastEditors: Jiachen Sun
LastEditTime: 2021-09-23 00:53:53
'''
import random
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
import torchvision
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import fourier_basis



parser = argparse.ArgumentParser(description='Fourier Analysis')
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--type", type=str, default='mag', help="cor type")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def generate_mask(f_c):
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.abs(np.sqrt((i-15.5) ** 2 + (j-15.5) ** 2)-f_c)**1.5)
    return mask

def generate_mask2():
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.abs(np.sqrt((i-15.5) ** 2 + (j-15.5) ** 2))**1.5)
    return mask
    
# MASK = generate_mask()
# MASK2 = generate_mask2()


if __name__ == "__main__":
    
    dataset_orig = get_dataset("cifar10", "test")
    for f_c in [1, 3, 5, 7, 9, 11, 13, 15]:
        all_data = []
        all_label = []
        for sev in [1,2,3]:
            
            plot = []
            labels = []
            c = [0.6,0.65,0.7,0.75,0.8][sev-1]
            d = [1.5,1.45,1.4,1.35,1.3][sev-1]
            e = [2,3,4,5,6][sev-1]
            f = [0.15,0.2,0.25][sev-1] 
            g = [0.4,0.5,0.6,0.7,0.8][sev-1] 
            basis = fourier_basis.generate_basis(e).cpu().numpy()
            for i in range(len(dataset_orig)):

                (x_orig, label) = dataset_orig[i]
                x_orig = x_orig.detach().numpy()

                x_orig_f = np.fft.fftshift(np.fft.fft2(x_orig))
                x_orig_f_abs = np.abs(x_orig_f)
                x_orig_f_ang = np.angle(x_orig_f)
                if args.type == 'abs_neg':
                    x_orig_f_abs *=  1 - np.random.rand(*x_orig_f_abs.shape) * c
                elif args.type == 'abs_pos':
                    x_orig_f_abs *=  1 + np.random.rand(*x_orig_f_abs.shape) * c
                elif args.type == 'angle':
                    x_orig_f_ang += (np.random.rand(*x_orig_f_abs.shape) - 0.5) * np.pi / d
                elif args.type == 'abs_2':
                    n_abs = (np.random.rand(*x_orig_f_abs.shape)) + f * MASK * MASK2
                    n_pha = np.random.uniform(*x_orig_f_ang.shape) * 2 * np.pi
                    n = np.zeros_like(x_orig_f)
                    n.real = n_abs * np.cos(n_pha)
                    n.imag = n_abs * np.sin(n_pha)
                    x_orig_f += n
                elif args.type == 'fourier':
                    n_abs = (np.random.rand(*x_orig_f_abs.shape)) + f * generate_mask(f_c) * x_orig_f_abs #/np.linalg.norm(x_orig_f_abs)
                    n_pha = np.random.uniform(*x_orig_f_ang.shape) * 2 * np.pi
                    n = np.zeros_like(x_orig_f)
                    n.real = n_abs * np.cos(n_pha)
                    n.imag = n_abs * np.sin(n_pha)
                    x_orig_f += n
                elif args.type == 'mixup':
                    j = random.randint(0,len(dataset_orig)-1)
                    x = dataset_orig[j][0]
                    x_f = np.fft.fftshift(np.fft.fft2(x.detach().numpy()))
                    x_f_abs = np.abs(x_f)
                    x_orig_f_abs = x_orig_f_abs * (1-g) + g * x_f_abs

                if args.type in ['abs_2','fourier']:
                    x_restored = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))
                else:  
                    x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
                    x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
                    
                    x_restored = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))

                plot.append(x_restored)
                labels.append(label)
                
            plot = np.clip((np.stack(plot) * 255).astype(int), 0,255)

            all_data.append(plot)
            all_label.append(labels)

        all_data = np.concatenate(all_data)
        all_label = np.concatenate(all_label)
        # print(plot)

        plot = np.transpose(plot, (0, 2, 3, 1))
        # labels = np.array(labels)

        print(all_data.shape, all_label.shape)
        os.makedirs('./data/CIFAR-10-F',exist_ok = True)
        
        np.save('./data/CIFAR-10-F/' + args.type + '_' + str(f_c) + '.npy',all_data)
        np.save('./data/CIFAR-10-F/label.npy',all_label)
        print(all_label[:10])

        plot = torch.FloatTensor(all_data[10000:10100])
        
        test_img = torchvision.utils.make_grid(plot/255., nrow = 10)
        torchvision.utils.save_image(
            test_img, "./test/filter/fourier_"+str(f_c)+".png", nrow = 10
        )
   
    # plt.close()