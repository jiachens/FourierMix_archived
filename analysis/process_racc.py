'''
Description: 
Autor: Jiachen Sun
Date: 2022-01-28 12:27:41
LastEditors: Jiachen Sun
LastEditTime: 2022-01-29 18:27:23
'''
import os
import argparse
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--model", type=str)
parser.add_argument("--clean", default=False, action='store_true')
parser.add_argument("--acr", default=False, action='store_true')
args = parser.parse_args()


# MODEL = ['resnet_18_rebuttal']
COR_H = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'pixelate', 'jpeg_compression']
COR_M = [ 'defocus_blur', 'frosted_glass_blur', 'motion_blur', 'zoom_blur', 'elastic']
COR_L = ['contrast','fog','snow','frost','brightness']
CLEAN = ['cifar']
SEV = ['1','2','3','4','5']
EPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def calculate2(corruptions,dir,clean=False):
    acr = 0
    for cor in corruptions:
        if clean:
            file_name = cor + '.out'
            output = os.path.join(dir,file_name)
            f = open(output,'r')
            # error = float(f.readlines()[-1].split(' ')[-2].split('=')[-1])
            # total_error += error
            acr = float(f.readlines()[-2].split(':')[-1].stripe())
            return acr
        else:
            for sev in SEV:
                file_name = cor + '_' + sev + '.out'
                output = os.path.join(dir,file_name)
                f = open(output,'r')
                # error = float(f.readlines()[-1].split(' ')[-2].split('=')[-1])
                # total_error += error
                acr += float(f.readlines()[-2].split(':')[-1].stripe())

    return acr / (len(corruptions) * len(SEV))

def calculate(corruptions,dir,eps,clean=False):

    total = 0
    total_correct = 0
    for cor in corruptions:
        if clean:
            file_name = cor + '.out'
            output = os.path.join(dir,file_name)
            f = open(output,'r')
            # error = float(f.readlines()[-1].split(' ')[-2].split('=')[-1])
            # total_error += error
            for line in f.readlines()[1:]:
                line = line.split('\t')
                # print(line)
                if len(line) == 6:
                    total += 1
                    if line[4] == '1' and float(line[3]) > eps:
                        total_correct += 1
        else:
            for sev in SEV:
                file_name = cor + '_' + sev + '.out'
                output = os.path.join(dir,file_name)
                f = open(output,'r')
                # error = float(f.readlines()[-1].split(' ')[-2].split('=')[-1])
                # total_error += error
                for line in f.readlines()[1:]:
                    line = line.split('\t')
                    # print(line)
                    if len(line) == 6:
                        total += 1
                        if line[4] == '1' and float(line[3]) > eps:
                            total_correct += 1

    return total_correct / total
            
def process(data_dir='./test/'):

    print("evaluating " + args.model + " ......")
    _dir = os.path.join(data_dir, args.model)
    if args.acr:
        if args.clean:
            print("Clean ACR :" )
            res.append(calculate(CLEAN,os.path.join(data_dir,'cifar10',args.model),True))
        else:
            print("High Frequency ACR :" )
            print(calculate2(COR_H,os.path.join(data_dir,'cifar10-c',args.model)))
            print("Mid Frequency ACR :" )
            print(calculate2(COR_M,os.path.join(data_dir,'cifar10-c',args.model)))
            print("Low Frequency ACR :" )
            print(calculate2(COR_L,os.path.join(data_dir,'cifar10-c',args.model)))

    else:  
        if args.clean:
            print("Clean RACC :" )
            res = []
            for eps in EPS:
                res.append(calculate(CLEAN,os.path.join(data_dir,'cifar10',args.model),eps,True))
            print(res)
        else:
            res_h = []
            res_m = []
            res_l = []
            print("High Frequency RACC :" )
            for eps in EPS:
                res_h.append(calculate(COR_H,os.path.join(data_dir,'cifar10-c',args.model),eps))
            print(res_h)
            print("Mid Frequency RACC :" )
            for eps in EPS:
                res_m.append(calculate(COR_M,os.path.join(data_dir,'cifar10-c',args.model),eps))
            print(res_m)
            print("Low Frequency RACC :" )
            for eps in EPS:
                res_l.append(calculate(COR_L,os.path.join(data_dir,'cifar10-c',args.model),eps))
            print(res_l)

if __name__ == '__main__':
    process()