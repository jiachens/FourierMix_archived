'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-13 16:53:22
LastEditors: Jiachen Sun
LastEditTime: 2021-07-14 21:41:08
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
from third_party import ttt_helper
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
import numpy as np
import torchvision



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
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
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--scheme', default='ga', type=str,
                    help='training schemes like gaussian augmentation')
parser.add_argument('--shared', default='layer2', type=str,
                    help='branching point')
parser.add_argument('--rotation_type', default='rand', type=str,
                    help='rotation type in SSL')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument("--no_normalize", default=True, action='store_false')
parser.add_argument("--severity", type=int, default=1, help="severity level to augment training using corruptions")

args = parser.parse_args()

### 

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train', scheme = args.scheme, severity=args.severity)
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    # model = get_architecture(args.arch, args.dataset,args.no_normalize)

    net, ext, head, ssh = ttt_helper.build_model(args)

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    parameters = list(net.parameters())+list(head.parameters())
    optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.arch in ['cifar_resnet110']:
        scheduler = MultiStepLR(optimizer,milestones=[100, 150],gamma=args.gamma)
    else:
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if args.arch in ['cifar_resnet110']:
        # for resnet110 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, net, ssh, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, net, criterion, args.noise_sd)
        test_loss_rot, test_acc_rot = test_ssl(test_loader, ssh, criterion, args.noise_sd)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
        
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(), 
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))


def train(loader: DataLoader, model: torch.nn.Module, model_ssl: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_rot = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model_ssl.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        if args.scheme in ['ttt_ga']:
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
        elif args.scheme in ['ttt_half_ga']:
            index = np.random.choice(inputs.shape[0],inputs.shape[0]//2)
            inputs[index] = inputs[index] + torch.randn_like(inputs[index], device='cuda') * noise_sd

        # print(model)
        # compute output

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if args.shared is not None:
            inputs_ssh, labels_ssh = ttt_helper.rotate_batch(inputs, args.rotation_type)
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            # print(inputs_ssh.shape, labels_ssh.shape)
            # if i == 0:
            #     test_img = torchvision.utils.make_grid(inputs_ssh, nrow = 33)
            #     torchvision.utils.save_image(
            #         test_img, "./test/test_ttt.png", nrow = 33
            #     )
            outputs_ssh = model_ssl(inputs_ssh)
            loss_ssh = criterion(outputs_ssh, labels_ssh)
            loss += loss_ssh
            

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        acc1_rot, _ = accuracy(outputs_ssh, labels_ssh, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        top1_rot.update(acc1_rot.item(), inputs_ssh.size(0))

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
                  'Acc@1_Rot {top1_rot.val:.3f} ({top1_rot.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1_rot=top1_rot, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            # compute output
            outputs = model(inputs)
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


def test_ssl(loader: DataLoader, model_ssl: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model_ssl.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            # compute output
            inputs_ssh, labels_ssh = ttt_helper.rotate_batch(inputs, args.rotation_type)
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            outputs_ssh = model_ssl(inputs_ssh)
            loss_ssh = criterion(outputs_ssh, labels_ssh)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(outputs_ssh, labels_ssh, topk=(1, 3))
            losses.update(loss_ssh.item(), inputs_ssh.size(0))
            top1.update(acc1.item(), inputs_ssh.size(0))
            top3.update(acc3.item(), inputs_ssh.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test Rotation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top3=top3))

        return (losses.avg, top1.avg)

if __name__ == "__main__":
    main()
