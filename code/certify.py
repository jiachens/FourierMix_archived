'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-09 00:21:36
LastEditors: Jiachen Sun
LastEditTime: 2021-06-14 22:48:45
'''
# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if args.dataset == "cifar10-c":
        # print(args.path)
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    elif args.dataset == "cifar10-c-bar":
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    else:
        dataset = get_dataset(args.dataset, args.split)
    
    total = 0
    total_correct = 0
    total_r = 0

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        if x.shape[0] != 3:
            x = x.permute(2,0,1)
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)
        total_correct += correct
        if correct == 1:
            total_r += radius

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        total += 1
            
    print("Accuracy: {}".format(total_correct/total), file=f, flush=True)
    print("Radius: {}".format(total_r/total_correct), file=f, flush=True)
    f.close()
