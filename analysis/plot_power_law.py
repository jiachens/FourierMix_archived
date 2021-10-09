'''
Description: 
Autor: Jiachen Sun
Date: 2021-10-07 22:19:25
LastEditors: Jiachen Sun
LastEditTime: 2021-10-07 22:55:34
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

def power_law(x,alpha):
    y = []
    for i in x:
        y.append(1./(i+1)**alpha)
    return y

x = np.linspace(0,16,num=128)
y1 = power_law(x,3)
y2 = power_law(x,4)
y3 = power_law(x,5)
y4 = power_law(x,6)

plt.figure(figsize=(12,3))

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

plt.grid( color = 'black',linestyle='-.',linewidth = 0.5)

l1=plt.plot(x,y1, label='$alpha$=3',markersize=9,lw=2)
l2=plt.plot(x,y2, label='$alpha$=4',markersize=9,lw=2)
l3=plt.plot(x,y3, label='$alpha$=5',markersize=9,lw=2)
l4=plt.plot(x,y4, label='$alpha$=6',markersize=9,lw=2)


plt.xlabel('x')
plt.ylabel('y') 
# plt.xlabel('Adversarial Budget $\epsilon$ (PGD-200)',fontsize=27.5)
# plt.ylabel('Adversarial Accuracy (%)',fontsize=27.5)
plt.legend()

# plt.xlim(0, 2001)
plt.ylim(0, 1)
# my_y_ticks = np.arange(0, 21, 4)
# my_x_ticks = ['0.0', '0.01', '0.025', '0.05', '0.075']
# plt.yticks(my_y_ticks)
# plt.xticks(x,my_x_ticks)
# plt.legend(fontsize=27.5, bbox_to_anchor=(1.1,1.0),borderaxespad = 0.)
plt.xticks(range(0, 16))#, color="red", rotation=45)
plt.yticks(range(0, 2))#, color="red", rotation=45)
plt.savefig('./test_power_law.png',dpi=300,bbox_inches='tight')