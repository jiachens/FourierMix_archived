'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-23 11:44:13
LastEditors: Jiachen Sun
LastEditTime: 2021-10-31 10:28:35
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Fourier Analysis -- Certified Accuracy Plot')
parser.add_argument("--path", type=str, help="path to folder")
args = parser.parse_args()

heatmap = np.zeros((31,31))
heatmap_1 = np.zeros((31,31))
heatmap_2 = np.zeros((31,31))

for i in range(961):
    row = i // 31
    col = i % 31
    try:    
        f = open(args.path + "/fourier_" + str(i) + ".out", "r")
        data = f.readlines()
        heatmap[row,col] = float(data[-4].split(' ')[-1])
        heatmap_1[row,col] = float(data[-3].split(' ')[-1])
        heatmap_2[row,col] = float(data[-2].split(' ')[-1])
        f.close()
    except:
        print(i)

ax = sns.heatmap(heatmap,
            cmap="jet",
            cbar=True,
            # cbar_kws={"ticks":[]},
            xticklabels=False,
            yticklabels=False,)
plt.savefig(args.path + '/certified_accuracy_hm.png',dpi=250,bbox_inches='tight')
# plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
plt.close()

ax = sns.heatmap(heatmap_1,
            cmap="jet",
            cbar=True,
            # cbar_kws={"ticks":[]},
            xticklabels=False,
            yticklabels=False,)
plt.savefig(args.path + '/correct_radius_hm.png',dpi=250,bbox_inches='tight')
# plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
plt.close()

ax = sns.heatmap(heatmap_2,
            cmap="jet",
            cbar=True,
            vmin = 0.0,
            vmax = 0.3,
            # cbar_kws={"ticks":[]},
            xticklabels=False,
            yticklabels=False,)
plt.savefig(args.path + '/incorrect_radius_hm.png',dpi=250,bbox_inches='tight')
# plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
plt.close()
