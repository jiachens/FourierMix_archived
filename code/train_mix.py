'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-21 21:25:03
LastEditors: Jiachen Sun
LastEditTime: 2021-07-22 16:35:28
'''
import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from PIL import ImageEnhance
import numpy as np
import torchvision
import random
from torchvision import transforms
from augment_and_mix import AugMixDataset 


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log')
parser.add_argument('--pre_path', type=str, help='folder to existing expert models')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--scheme', default='ga', type=str,
                    help='training schemes like gaussian augmentation')
# parser.add_argument('--expert', default='autocontrast', type=str,
#                     help='augmix expert')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument("--no_normalize", default=True, action='store_false')
parser.add_argument("--jsd", default=False, action='store_true')

args = parser.parse_args()

def loadcheckpoint(path):
    checkpoint = torch.load(path)
    try:
        base_classifier = get_architecture("cifar_resnet110", args.dataset, args.no_normalize)
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
    return base_classifier

EXPERT = ['autocontrast','equalize','solarize','posterize']

def main():
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    expert_model = [loadcheckpoint(args.pre_path + '/expert_half_ga_no_normalize_' + path + '/checkpoint.pth.tar') for path in EXPERT]
    para = []
    for expert in expert_model:
        para += list(expert.fc.parameters()) #list(expert.layer3.parameters())
    # para_total = [para + list(expert.layer3.parameters()) + list(expert.fc.parameters()) for expert in expert_model]

    train_dataset = get_dataset(args.dataset, 'train', scheme = args.scheme)
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_data =  AugMixDataset(train_dataset, preprocess, 3, 1., not(args.jsd)) #MixDataset(train_dataset, preprocess, args.expert, 'train', not(args.jsd))
    # test_data = MixDataset(test_dataset, preprocess, args.expert, 'test')
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)                             
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset,args.no_normalize)

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(para + list(model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.arch in ['cifar_resnet110','cifar_resnet20_4']:
        scheduler = MultiStepLR(optimizer,milestones=[100, 150],gamma=args.gamma)
    else:
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if args.arch in ['cifar_resnet110','cifar_resnet20_4']:
        # for resnet110 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, expert_model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, model, expert_model, criterion, args.noise_sd)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
        
        scheduler.step(epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

        for i,expert in enumerate(EXPERT):
            torch.save({
                'arch': "cifar_resnet110",
                'state_dict': expert_model[i].state_dict(),
            }, os.path.join(args.outdir, expert + '_checkpoint.pth.tar'))


def train(loader: DataLoader, model: torch.nn.Module, expert_model, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    for expert in expert_model:
        expert.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        if args.scheme in ['expert_ga']:
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
        elif args.scheme in ['expert_half_ga']:
            index = np.random.choice(inputs.shape[0],inputs.shape[0]//2)
            inputs[index] = inputs[index] + torch.randn_like(inputs[index], device='cuda') * noise_sd


        # if i == 0:
        #     test_img = torchvision.utils.make_grid(inputs, nrow = 16)
        #     torchvision.utils.save_image(
        #         test_img, "./test/test_"+args.scheme+"_"+str(noise_sd)+"_"+str(args.expert)+".png", nrow = 16
        #     )

        expert_output = [expert(inputs) for expert in expert_model]
        weight_output = torch.unsqueeze(model(inputs),dim=-1)  
        expert_output = torch.stack(expert_output,dim=1)

        outputs = torch.mul(expert_output,weight_output)
        outputs = torch.sum(outputs,dim=1)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)



def test(loader: DataLoader, model: torch.nn.Module, expert_model, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    for expert in expert_model:
        expert.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            expert_output = [expert(inputs) for expert in expert_model]
            weight_output = torch.unsqueeze(model(inputs),dim=-1)  
            expert_output = torch.stack(expert_output,dim=1)

            outputs = torch.mul(expert_output,weight_output)
            outputs = torch.sum(outputs,dim=1)

            
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)



if __name__ == "__main__":
    main()