'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-09 00:21:36
LastEditors: Jiachen Sun
LastEditTime: 2021-07-15 14:43:13
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
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--batch", type=int, default=64, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
parser.add_argument("--no_normalize", default=True, action='store_false')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if __name__ == "__main__":
    # load the base classifier
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
                    name = key[7:]  # remove 'module.'
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

    # create the smooothed classifier g
    # smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    # prepare output file
    dir = args.outfile[:-(len(args.outfile.split('/')[-1])+1)]
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok = True)
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

    img = torch.Tensor(dataset.data[:args.batch])
    if img.shape[1] != 3:
        img = img.permute(0,3,1,2)

    test_data = torch.utils.data.DataLoader(dataset, batch_size=args.batch, pin_memory=True, num_workers=4)
    
    model = BoundedModule(base_classifier, torch.empty_like(img), bound_opts={"conv_mode": "patches"})
    ptb = PerturbationLpNorm(norm=2, eps=0.2)
    
    total = 0
    total_correct = 0
    base_total_correct = 0
    total_r = 0
    total_r_with_incorrect = 0

    for i, (x, label) in enumerate(test_data):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        # (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        if x.shape[1] != 3:
            x = x.permute(0,3,1,2)
        x = BoundedTensor(x, ptb)
        prediction = model(x)
        after_time = time()
        p_label = torch.argmax(prediction, dim=1).cpu().numpy()
        # correct = int(prediction == label)
        ## Step 5: Compute bounds for final output
        for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
            lb, ub = model.compute_bounds(x=(x,), method=method.split()[0])
            lb = lb.detach().cpu().numpy()
            ub = ub.detach().cpu().numpy()
            print("Bounding method:", method)
            for i in range(args.batch):
                print("Image {} top-1 prediction {} ground-truth {}".format(i, p_label[i], label[i]))
                for j in range(10):
                    indicator = '(ground-truth)' if j == label[i] else ''
                    print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                        j=j, l=lb[i][j], u=ub[i][j], ind=indicator))
            print()

    #     total_correct += correct

    #     if correct == 1:
    #         total_r += radius
        
    #     total_r_with_incorrect += radius

    #     time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    #     print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
    #         i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    #     total += 1
    
    # print("Empirical Accuracy: {}".format(base_total_correct/total), file=f, flush=True)
    # print("Certified Accuracy: {}".format(total_correct/total), file=f, flush=True)
    # print("Correct Radius: {}".format(total_r/total_correct), file=f, flush=True)
    # print("Correct Radius (with 0 for incorrect): {}".format(total_r/total), file=f, flush=True)
    # print("Radius: {}".format(total_r_with_incorrect/total), file=f, flush=True)
    f.close()
