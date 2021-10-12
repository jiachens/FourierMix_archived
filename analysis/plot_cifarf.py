'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-12 14:38:44
LastEditors: Jiachen Sun
LastEditTime: 2021-10-12 15:00:50
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator


for alpha in [0.1, 0.5, 1, 2, 3]:
    name = ['cifar10_half_ga_no_normalize_0.25','test_ga_consistency_0.25','test_auto_jsd_2_0.25',
            'augmix_half_ga_0.25','test_fourier_half_ga_13_0.25','cifar10_fourier_consistency_lbd2_40_0.25']
    y = [[],[],[],[],[],[]]

    for j in range(6):
        for i in range(1,17):
            c_r = 0
            for sev in ['1','2','3']:
                f = open(os.path.join('../test/cifar10-f-new', name[j] ,str(i) + '_' + str(alpha) + '_' + sev + '.out'))
                lines = f.readlines()
                print(name[j],i,alpha)
                print(lines[-2])
                c_r += float(lines[-2].split(':')[-1].strip())
                f.close()
            y[j].append(c_r / 3.)

    x = np.linspace(1,16,num=16)

    plt.figure(figsize=(12,6))

    ax = plt.gca()#获取边框
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(15))

    plt.grid( color = 'black',linestyle='-.',linewidth = 0.25)

    l1=plt.plot(x,y[0], marker='o', label='Gaussain',markersize=9,lw=2)
    l2=plt.plot(x,y[1], marker='o',label='Gaussian+JSD',markersize=9,lw=2)
    l3=plt.plot(x,y[2],marker='o', label='AutoAugment+JSD',markersize=9,lw=2)
    l4=plt.plot(x,y[3],marker='o', label='AugMix+JSD',markersize=9,lw=2)
    l5=plt.plot(x,y[4], marker='o',label='FourierMix+JSD',markersize=9,lw=2)
    l6=plt.plot(x,y[5],marker='o', label='FourierMix+HCR',markersize=9,lw=2)



    plt.xlabel('center frequency $f_c$',fontsize=20,)
    plt.ylabel('Average Certified Radius (ACR)',fontsize=20,)
    plt.title(r'$\alpha$='+str(alpha),fontsize=20,)  
    plt.legend(fontsize=15)

    plt.ylim(0.2, 0.55)

    plt.xticks(np.linspace(1,16,num=16),fontsize=15)#, color="red", rotation=45)
    plt.yticks(np.linspace(0.2,0.55,num=6),fontsize=15)#, color="red", rotation=45)
    plt.savefig('./analysis_'+str(alpha)+'.png',dpi=300,bbox_inches='tight')

    plt.close()