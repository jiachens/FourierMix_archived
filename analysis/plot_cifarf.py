'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-12 14:38:44
LastEditors: Jiachen Sun
LastEditTime: 2021-11-16 14:27:58
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

sevs = ['1','2','3']
for alpha in [0.5, 1, 2, 3]:
    name = ['cifar10_half_ga_no_normalize_0.25','test_ga_consistency_0.25','test_auto_jsd_2_0.25',
            'augmix_half_ga_0.25','cifar10_augmix_hcr_40_0.25','test_fourier_half_ga_13_0.25','cifar10_fourier_consistency_lbd2_40_0.25']
    y = [[],[],[],[],[],[],[]]
    
    for j in range(7):
        for i in range(1,17):
            c_r = 0
            
            for sev in sevs:
                f = open(os.path.join('../test/cifar10-f-new', name[j] ,str(i) + '_' + str(alpha) + '_' + sev + '.out'))
                lines = f.readlines()
                # print(name[j],i,alpha,sev)
                # print(lines[-2])
                c_r += float(lines[-2].split(':')[-1].strip())
                f.close()
            y[j].append(c_r/len(sevs))
            # print(y[j])

    x = np.linspace(1,16,num=16)

    plt.figure(figsize=(12,10))

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(15))

    plt.grid( color = 'black',linestyle='-.',linewidth = 0.25)

    l1=plt.plot(x,y[0], marker='^', label='Gaussain',markersize=12,lw=3, color='#845EC2')
    l2=plt.plot(x,y[1], marker='^',label='+JSD',markersize=12,lw=3,color='#4B4453')
    l3=plt.plot(x,y[2],marker='^', label='+AutoAugment+JSD',markersize=12,lw=3,color='#B0A8B9')
    l4=plt.plot(x,y[3],marker='^', label='+AugMix+JSD',markersize=12,lw=3,color='#C34A36')
    l41=plt.plot(x,y[4],marker='^', label='+AugMix+HCR',markersize=12,lw=3,color='#4AA233')
    l5=plt.plot(x,y[5], marker='^',label='+FourierMix+JSD',markersize=12,lw=3,color='#FF8066')
    l6=plt.plot(x,y[6],marker='^', label='+FourierMix+HCR',markersize=12,lw=3,color='#4E8397')

    for i in range(len(y)):
        print(name[i],np.mean(y[i]))
    print('\n')

    plt.xlabel('Center Frequency $f_c$',fontsize=25,)
    plt.ylabel('Average Certified Radius (ACR)',fontsize=25,)
    # plt.title(r'$\alpha$='+str(alpha),fontsize=20,)  
    plt.legend(fontsize=25,loc=4,ncol=1)

    plt.xticks(np.linspace(1,16,num=16),fontsize=25)#, color="red", rotation=45)
    plt.yticks(np.linspace(0,1,21),fontsize=25)#, color="red", rotation=45)
    plt.ylim(np.min(y[0])-0.025, np.max(y[6])+0.025) 

    plt.savefig('./analysis_'+str(alpha)+'.png',dpi=300,bbox_inches='tight')

    plt.close()