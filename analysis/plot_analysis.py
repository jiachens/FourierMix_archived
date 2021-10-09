'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-08 13:46:17
LastEditors: Jiachen Sun
LastEditTime: 2021-10-08 14:55:57
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator


x = np.linspace(2,32,num=16)
y1 = [0.3387,0.2805,0.2416,0.2401,0.2620,0.2672,0.2922,0.3096,0.3223,0.3434,0.3649,0.3785,0.3830,0.3980,0.4060,0.4104]
y2 = [0.3948,0.3416,0.3385,0.3609,0.3770,0.4046,0.4215,0.4402,0.4463,0.4713,0.4808,0.4818,0.4926,0.5020,0.5048,0.5093]
y3 = [0.3419,0.2935,0.2557,0.2440,0.2357,0.2508,0.2535,0.2592,0.2736,0.2856,0.3022,0.3181,0.3327,0.3425,0.3572,0.3591]
y4 = [0.3887,0.3194,0.3033,0.2869,0.3026,0.3436,0.3489,0.3673,0.3854,0.4026,0.4177,0.4379,0.4510,0.4531,0.4621,0.4733]
y5 = [0.3895,0.3202,0.3043,0.3170,0.3348,0.3642,0.3907,0.4146,0.4206,0.4360,0.4429,0.4568,0.4664,0.4675,0.4792,0.4838]
y6 = [0.4087,0.3507,0.3395,0.3708,0.3820,0.4049,0.4229,0.4577,0.4685,0.4703,0.4809,0.4840,0.4983,0.5058,0.5040,0.5098]

plt.figure(figsize=(12,6))

ax = plt.gca()#获取边框
# ax.spines['top'].set_color('red')  
# ax.spines['right'].set_color('none')  
# ax.set_xticks = [0,20,50,100,200,500,1000,2000]
# ax.set_xticklabels = [0,20,50,100,200,500,1000,2000]
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
# ax.set_xscale("log")
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_major_locator(MultipleLocator(15))

plt.grid( color = 'black',linestyle='-.',linewidth = 0.25)

l1=plt.plot(x,y1, marker='o', label='Gaussain',markersize=9,lw=2)
l2=plt.plot(x,y2, marker='o',label='Gaussian+Cons',markersize=9,lw=2)
l3=plt.plot(x,y3,marker='o', label='AutoAugment+JSD',markersize=9,lw=2)
l4=plt.plot(x,y4,marker='o', label='AugMix+JSD',markersize=9,lw=2)
l5=plt.plot(x,y5, marker='o',label='FourierMix+JSD',markersize=9,lw=2)
l6=plt.plot(x,y6,marker='o', label='FourierMix+HCR',markersize=9,lw=2)



plt.xlabel('Bandwidth',fontsize=20,)
plt.ylabel('Average Certified Radius (ACR)',fontsize=20,) 
# plt.xlabel('Adversarial Budget $\epsilon$ (PGD-200)',fontsize=27.5)
# plt.ylabel('Adversarial Accuracy (%)',fontsize=27.5)
plt.legend(fontsize=15)

# plt.xlim(0, 2001)
plt.ylim(0.2, 0.55)
# my_y_ticks = np.arange(0, 21, 4)
# my_x_ticks = ['0.0', '0.01', '0.025', '0.05', '0.075']
# plt.yticks(my_y_ticks)
# plt.xticks(x,my_x_ticks)
# plt.legend(fontsize=27.5, bbox_to_anchor=(1.1,1.0),borderaxespad = 0.)
plt.xticks(np.linspace(2,32,num=16),fontsize=15)#, color="red", rotation=45)
plt.yticks(np.linspace(0.2,0.55,num=6),fontsize=15)#, color="red", rotation=45)
plt.savefig('./analysis_low.png',dpi=300,bbox_inches='tight')