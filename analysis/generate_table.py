'''
Description: 
Autor: Jiachen Sun
Date: 2021-08-18 16:11:26
LastEditors: Jiachen Sun
LastEditTime: 2021-10-30 01:54:09
'''

import argparse
import os
import numpy as np

C = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
]

C_BAR = [
    'blue_noise_sample',
    'checkerboard_cutout',
    'inverse_sparkles',
    'lines',
    'ripple',
    'brownish_noise',
    'circular_motion_blur',
    'pinch_and_twirl',
    'sparkles',
    'transverse_chromatic_abberation'
]

IMG_C_BAR = [
    'blue_noise_sample',
    'checkerboard_cutout',
    'inverse_sparkles',
    'perlin_noise',
    'cocentric_sine_waves',
    # 'brownish_noise',
    'plasma_noise',
    'caustic_refraction',
    'sparkles',
    'single_frequency_greyscale'
]



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('path', type=str)
parser.add_argument("dataset", choices=['cifar10-c','cifar10-c-bar','imagenet-c-bar'], help="which dataset")
args = parser.parse_args()

f_w = open(os.path.join(args.path,'output.txt'),'w')

if args.dataset == 'cifar10-c':
    for cor in C:
        f_w.write(cor+'\n')
        f_w.write('Severity EmpAcc AvgAcc AvgRadius ACR\n')
        avg = []
        for sev in ['1','2','3','4','5']:
            f = open(os.path.join(args.path,cor + '_' + sev + '.out'))
            lines = f.readlines()
            emp_acc = lines[-5].split(':')[-1].strip()
            cer_acc = lines[-4].split(':')[-1].strip()
            r = lines[-3].split(':')[-1].strip()
            c_r = lines[-2].split(':')[-1].strip()
            f_w.write(sev + ' ' + emp_acc + ' ' + cer_acc + ' ' + r[:6] + ' ' + c_r[:6]  +'\n')
            avg.append(float(c_r))
            f.close()
        f_w.write('Avg ACR: ' + str(np.mean(avg)) +'\n')

elif args.dataset == 'cifar10-c-bar':
    for cor in C_BAR:
        f_w.write(cor+'\n')
        f_w.write('Severity EmpAcc AvgAcc AvgRadius ACR\n')
        avg = []
        for sev in ['1','2','3','4','5']:
            f = open(os.path.join(args.path,cor + '_' + sev + '.out'))
            lines = f.readlines()
            emp_acc = lines[-5].split(':')[-1].strip()
            cer_acc = lines[-4].split(':')[-1].strip()
            r = lines[-3].split(':')[-1].strip()
            c_r = lines[-2].split(':')[-1].strip()
            f_w.write(sev + ' ' + emp_acc + ' ' + cer_acc + ' ' + r[:6]  + ' ' + c_r[:6] +'\n')
            avg.append(float(c_r))
            f.close()
        f_w.write('Avg ACR: ' + str(np.mean(avg)) +'\n')

elif args.dataset == 'imagenet-c-bar':
    for cor in IMG_C_BAR:
        f_w.write(cor+'\n')
        f_w.write('Severity EmpAcc AvgAcc AvgRadius ACR\n')
        avg = []
        for sev in ['1','2','3','4','5']:
            f = open(os.path.join(args.path,cor + '_' + sev + '.out'))
            lines = f.readlines()
            emp_acc = lines[-5].split(':')[-1].strip()
            cer_acc = lines[-4].split(':')[-1].strip()
            r = lines[-3].split(':')[-1].strip()
            c_r = lines[-2].split(':')[-1].strip()
            f_w.write(sev + ' ' + emp_acc + ' ' + cer_acc + ' ' + r[:6]  + ' ' + c_r[:6] +'\n')
            avg.append(float(c_r))
            f.close()
        f_w.write('Avg ACR: ' + str(np.mean(avg)) +'\n')

f_w.close()