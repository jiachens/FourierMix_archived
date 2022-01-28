'''
Description: 
Autor: Jiachen Sun
Date: 2022-01-28 12:27:41
LastEditors: Jiachen Sun
LastEditTime: 2022-01-28 16:44:06
'''
import os
import argparse
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--model", type=str)
parser.add_argument("--clean", default=False, action='store_true')
args = parser.parse_args()


# MODEL = ['resnet_18_rebuttal']
COR_H = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'pixelate', 'jpeg_compression']
COR_M = [ 'defocus_blur', 'frosted_glass_blur', 'motion_blur', 'zoom_blur', 'elastic']
COR_L = ['contrast','fog','snow','frost','brightness']
CLEAN = ['cifar10']
SEV = ['1','2','3','4','5']

def calculate(corruptions,dir,clean):

    total = 0
    total_correct = 0
    eps = 0.25
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
    if args.clean:
        print("Clean RACC :" )
        print(calculate(CLEAN,os.path.join(data_dir,'cifar10',args.model),True))
    else:
        print("High Frequency RACC :" )
        print(calculate(COR_H,os.path.join(data_dir,'cifar10-c',args.model)))
        print("Mid Frequency RACC :" )
        print(calculate(COR_M,os.path.join(data_dir,'cifar10-c',args.model)))
        print("Low Frequency RACC :" )
        print(calculate(COR_L,os.path.join(data_dir,'cifar10-c',args.model)))

if __name__ == '__main__':
    process()