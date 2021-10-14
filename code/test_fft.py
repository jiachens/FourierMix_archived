'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-12 22:45:00
LastEditors: Jiachen Sun
LastEditTime: 2021-10-13 21:48:01
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

def generate_mask(f_c,alpha):
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.abs(np.maximum(np.abs(i-15),np.abs(j-15))-f_c)+1.0)**alpha
    # mask /= np.linalg.norm(mask)
    # mask[mask > 0.5] = 1
    ax = sns.heatmap(mask,
            cmap="jet",
            cbar=True,
            vmin = 0.0,
            vmax = 1.0,
            # cbar_kws={"ticks":[]},
            xticklabels=False,
            yticklabels=False,)
    plt.savefig('./test/vis3/'+str(f_c)+'.png',dpi=250,bbox_inches='tight')
    # plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
    plt.close()
        
    return mask

def generate_mask2():
    mask = np.ones((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j] = 1/(np.abs(np.sqrt((i-16) ** 2 + (j-16) ** 2))+0.25)**1.5
    return mask
    
# MASK = generate_mask()
MASK2 = generate_mask2()


if __name__ == "__main__":
    
    dataset_orig = get_dataset("cifar10", "test")
    for alpha in [3]:
        for f_c in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            all_data = []
            all_label = []
            for sev in [1,2,3]:
                
                plot = []
                labels = []
                c = [0.6,0.65,0.7,0.75,0.8][sev-1]
                d = [1.5,1.45,1.4,1.35,1.3][sev-1]
                e = [2,3,4,5,6][sev-1]
                f = [0.5,1,1.5][sev-1] 
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
                        n_abs = (np.random.rand(*x_orig_f_abs.shape)) + f * generate_mask(f_c,alpha) * MASK2
                        n_pha = np.random.uniform(*x_orig_f_ang.shape) * 2 * np.pi
                        n = np.zeros_like(x_orig_f)
                        n.real = n_abs * np.cos(n_pha)
                        n.imag = n_abs * np.sin(n_pha)
                        x_orig_f += n
                    elif args.type == 'fourier':
                        n_abs = (np.random.normal(*x_orig_f_abs.shape)) + f * np.minimum(np.maximum(x_orig_f_abs,20),200) * generate_mask(f_c,alpha)
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
                        # delta = x_restored - x_orig
                        # print(np.max(delta))
                        # delta = delta / np.linalg.norm(delta) * f
                        # print(np.max(delta))
                        # x_restored = delta + x_orig
                    else:  
                        x_orig_f.real = x_orig_f_abs * np.cos(x_orig_f_ang)
                        x_orig_f.imag = x_orig_f_abs * np.sin(x_orig_f_ang)
                        
                        x_restored = np.abs(np.fft.ifft2(np.fft.ifftshift(x_orig_f)))

                    plot.append(x_restored)
                    labels.append(label)
                    if i == 99:
                        break
                    
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

            plot = torch.FloatTensor(all_data[200:300])
            
            test_img = torchvision.utils.make_grid(plot/255., nrow = 10)
            torchvision.utils.save_image(
                test_img, "./test/vis3/" + args.type + '_' + str(f_c) + '_' + str(alpha) + '_' + str(sev) + ".png", nrow = 10
            )
   
    # plt.close()