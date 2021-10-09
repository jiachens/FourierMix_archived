'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-08 15:02:39
LastEditors: Jiachen Sun
LastEditTime: 2021-10-08 21:34:28
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator


x = np.linspace(0,32,num=16)
y1 = [0.4119,0.4138,0.4150,0.4193,0.4270,0.4309,0.4384,0.4431,0.4474,0.4500,0.4548,0.4563,0.4587,0.4583,0.4596,0.4605]
y2 = [0.5060,0.5141,0.5172,0.5203,0.5221,0.5218,0.5262,0.5302,0.5297,0.5322,0.5332,0.5341,0.5347,0.5351,0.5349,0.5351]
y3 = [0.3658,0.3664,0.3691,0.3744,0.3777,0.3867,0.3882,0.3919,0.3993,0.4048,0.4110,0.4131,0.4162,0.4182,0.4201,0.4209]
y4 = [0.4784,0.4726,0.4831,0.4818,0.4907,0.4935,0.4963,0.5047,0.5049,0.5089,0.5119,0.5141,0.5152,0.5154,0.5154,0.5164]
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
plt.savefig('./analysis_high.png',dpi=300,bbox_inches='tight')