'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-13 22:48:22
LastEditors: Jiachen Sun
LastEditTime: 2021-07-15 17:01:06
'''
# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from architectures import get_architecture,ARCHITECTURES
from third_party import ttt_helper
import numpy as np


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--path", type=str, help="path to dataset")
parser.add_argument("--corruption", type=str, default="fog", help="corruption type when using cifar10-c")
parser.add_argument("--severity", type=int, default=1, help="severity level when using cifar10-c")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--batch_size", type=int, default=500, help="batch size for TTT")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--niter", type=int, default=10, help="number of steps for adaptation")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--gpu", type=str, default='0', help="which GPU to use")
parser.add_argument("--no_normalize", default=True, action='store_false')
parser.add_argument('--shared', default='layer2', type=str,
                    help='branching point')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


def adapt(data,arch,dir,net,ext,ssh):
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    if True:
        ssh.eval()
        ext.train()
        optimizer_ssh = optim.SGD(ext.parameters(), lr=0.001)
    # else:
    #     ssh.train()
    #     optimizer_ssh = optim.SGD(ssh.parameters(), lr=0.001)
    index = np.random.choice(len(data),args.batch_size,replace=False)
    inputs = [data[index[j]][0].permute(2,0,1) for j in range(args.batch_size)]
    inputs = torch.stack(inputs).cuda()
    inputs += torch.randn_like(inputs, device='cuda') * args.sigma
    inputs_ssh, labels_ssh = ttt_helper.rotate_batch(inputs, 'rand')
    inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
    
    for iteration in range(args.niter):
        
        optimizer_ssh.zero_grad()
        outputs_ssh = ssh(inputs_ssh)
        loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
        loss_ssh.backward()
        optimizer_ssh.step()
    print("Adaptation Done ...")
    torch.save({
        'arch': args.arch,
        'state_dict': net.state_dict(), 
        'optimizer': optimizer_ssh.state_dict(),
    }, os.path.join(dir, 'ttt_checkpoint_{}_{}.pth.tar'.format(args.corruption, args.severity)))
    return net

# def test(model, image, label):
# 	model.eval()
# 	inputs = ttt_helper.te_transforms(image).unsqueeze(0)
# 	with torch.no_grad():
# 		outputs = model(inputs.cuda())
# 		_, predicted = outputs.max(1)
# 		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
# 	correctness = 1 if predicted.item() == label else 0
# 	return correctness, confidence


if __name__ == "__main__":
    # load the base classifier
    net, ext, head, ssh = ttt_helper.build_model(args)
    checkpoint = torch.load(args.base_classifier)
    net.load_state_dict(checkpoint['state_dict'])
    head.load_state_dict(checkpoint['head'])
    dir = args.outfile[:-(len(args.outfile.split('/')[-1])+1)]
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok = True)
    # iterate through the dataset
    if args.dataset == "cifar10-c":
        # print(args.path)
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    elif args.dataset == "cifar10-c-bar":
        dataset = get_dataset(args.dataset, None, args.path, args.corruption, args.severity)
    else:
        dataset = get_dataset(args.dataset, args.split)

    net = adapt(dataset,checkpoint['arch'],dir,net,ext,ssh)
    net.eval()
    head.eval()
    # create the smooothed classifier g
    smoothed_classifier = Smooth(net, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    
    
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
        if x.shape[0] != 3:
            x = x.permute(2,0,1)
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
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        total += 1
    
    print("Empirical Accuracy: {}".format(base_total_correct/total), file=f, flush=True)
    print("Certified Accuracy: {}".format(total_correct/total), file=f, flush=True)
    print("Correct Radius: {}".format(total_r/total_correct), file=f, flush=True)
    print("Correct Radius (with 0 for incorrect): {}".format(total_r/total), file=f, flush=True)
    print("Radius: {}".format(total_r_with_incorrect/total), file=f, flush=True)

    f.close()