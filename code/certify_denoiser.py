'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-26 23:58:58
LastEditors: Jiachen Sun
LastEditTime: 2021-07-26 23:58:59
'''
# evaluate a smoothed classifier on a dataset
from architectures import get_architecture, IMAGENET_CLASSIFIERS
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
from time import time

import argparse
import datetime
import os
import torch

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument("--no_normalize", default=True, action='store_false')
args = parser.parse_args()


if __name__ == "__main__":
    # load the base classifier
    if args.base_classifier in IMAGENET_CLASSIFIERS:
        assert args.dataset == 'imagenet'
        # loading pretrained imagenet architectures
        base_classifier = get_architecture(args.base_classifier ,args.dataset)
    else:
        checkpoint = torch.load(args.base_classifier)
        try:
            base_classifier = get_architecture(checkpoint["arch"], args.dataset, args.no_normalize)
            base_classifier.load_state_dict(checkpoint['state_dict'])
        except:
            base_classifier = get_architecture("cifar_resnet110", args.dataset, args.no_normalize)
            # print(checkpoint['model_state_dict'].keys())
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            if 'model_state_dict' in checkpoint.keys():
                for key, val in checkpoint['model_state_dict'].items():
                    # print(key)
                    if key[:6] == 'module':
                        name = key[7:]  # remove 'module.'q
                    else:
                        name = key
                    new_state_dict[name] = val
            else:
                for key, val in checkpoint['state_dict'].items():
                    # print(key)
                    if key[:6] == 'module':
                        name = key[7:]  # remove 'module.'
                    else:
                        name = key
                    new_state_dict[name] = val
            base_classifier.load_state_dict(new_state_dict)

    if args.denoiser != '':
        checkpoint = torch.load(args.denoiser)
        if "off-the-shelf-denoiser" in args.denoiser:
            denoiser = get_architecture('orig_dncnn', args.dataset)
            denoiser.load_state_dict(checkpoint)
        else:
            denoiser = get_architecture(checkpoint['arch'] ,args.dataset)
            denoiser.load_state_dict(checkpoint['state_dict'])
                
        base_classifier = torch.nn.Sequential(denoiser, base_classifier)

    base_classifier = base_classifier.eval().cuda()

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    if not os.path.exists(args.outfile.split('sigma')[0]):
        os.makedirs(args.outfile.split('sigma')[0])

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
    f.close()

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
    base_total_correct = 0
    total_r = 0
    total_r_with_incorrect = 0
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

        base_prediction = smoothed_classifier.base_predict(x)
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)
        base_correct = int(base_prediction == label)
        total_correct += correct
        base_total_correct += base_correct

        if correct == 1:
            total_r += radius
        
        total_r_with_incorrect += radius

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        f = open(args.outfile, 'a')
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), flush=True)

        total += 1

        f.close()

    f = open(args.outfile, 'a')
    
    print("Empirical Accuracy: {}".format(base_total_correct/total), file=f, flush=True)
    print("Certified Accuracy: {}".format(total_correct/total), file=f, flush=True)
    print("Correct Radius: {}".format(total_r/total_correct), file=f, flush=True)
    print("Correct Radius (with 0 for incorrect): {}".format(total_r/total), file=f, flush=True)
    print("Radius: {}".format(total_r_with_incorrect/total), file=f, flush=True)
    f.close()

    print("Empirical Accuracy: {}".format(base_total_correct/total), flush=True)
    print("Certified Accuracy: {}".format(total_correct/total), flush=True)
    print("Correct Radius: {}".format(total_r/total_correct), flush=True)
    print("Correct Radius (with 0 for incorrect): {}".format(total_r/total), flush=True)
    print("Radius: {}".format(total_r_with_incorrect/total), flush=True)