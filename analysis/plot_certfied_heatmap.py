'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-23 11:44:13
LastEditors: Jiachen Sun
LastEditTime: 2021-06-23 12:39:56
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
for i in range(961):
    row = i // 31
    col = i % 31
    f = open(args.path + "fourier_" + str(i) + ".out", "r")
    heatmap[row,col] = float(f.readlines()[-4].split(' ')[-1])
    f.close()

ax = sns.heatmap(heatmap,
            cmap="jet",
            cbar=True,
            # cbar_kws={"ticks":[]},
            xticklabels=False,
            yticklabels=False,)
plt.savefig('./test/certified_accuracy_hm.png',dpi=250,bbox_inches='tight')
# plt.savefig('./figures/fourier_analysis/' + args.corruption +  '_' + args.severity + '.png',dpi=250,bbox_inches='tight')    
plt.close()