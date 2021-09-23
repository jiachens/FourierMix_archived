'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-30 16:33:35
LastEditors: Jiachen Sun
LastEditTime: 2021-09-22 21:02:09
'''
import time
import setGPU
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
import torchvision
import cifar10_c
import cifar10_c_bar
import cifar100_c
from architectures import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
import consistency
from fourier_augment import FourierDataset

parser = argparse.ArgumentParser(description='PyTorch AugMix Training')
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
parser.add_argument("--no_normalize", default=True, action='store_false')
parser.add_argument("--path", type=str, help="path to cifar10-c dataset")

parser.add_argument("--lbd1", type=int,default=10)
parser.add_argument("--lbd2", type=int,default=10)

args = parser.parse_args()


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'frosted_glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic', 'pixelate',
    'jpeg_compression'
]


def main():
    epochs = args.epochs
    k = 0
    p = 0
    js_loss = True
    batch_size = args.batch
    new_loss = True


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok = True)

    f = open(file=os.path.join(args.outdir,'log.txt'),mode='a')

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    pin_memory = (args.dataset == "imagenet")

    # load data
    train_dataset = get_dataset(args.dataset, 'train', scheme = args.scheme)
    test_data = get_dataset(args.dataset, 'test')

    train_data = FourierDataset(train_dataset, k, p, not(js_loss))

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)

    # 2. model
    model = get_architecture(args.arch, args.dataset, args.no_normalize)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.arch in ['cifar_resnet110']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150],gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    model = nn.DataParallel(model).to(device)
    cudnn.benchmark = True

    # training model with cifar100
    # model.train()
    losses = []
    t = time.time()

    for epoch in range(epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            if new_loss:
                bs = images[0].size(0)
                images[0],images[1],images[2] = images[0].to(device),images[1].to(device),images[2].to(device)
                images_0_0 = images[0] + torch.randn_like(images[0], device='cuda') * args.noise_sd
                images_0_1 = images[0] + torch.randn_like(images[0], device='cuda') * args.noise_sd
                images_1_0 = images[1] + torch.randn_like(images[1], device='cuda') * args.noise_sd
                images_1_1 = images[1] + torch.randn_like(images[1], device='cuda') * args.noise_sd
                images_2_0 = images[2] + torch.randn_like(images[2], device='cuda') * args.noise_sd
                images_2_1 = images[2] + torch.randn_like(images[2], device='cuda') * args.noise_sd

                images_cat = torch.cat([images_0_0,images_0_1,images_1_0,images_1_1,images_2_0,images_2_1], dim = 0).to(device)
                
                targets = targets.to(device)

                logits = model(images_cat)
                logits_orig_0, logits_orig_1,logits_aug1_0, logits_aug1_1,logits_aug2_0, logits_aug2_1 = logits[:bs], logits[bs:2*bs], logits[2*bs:3*bs],logits[3*bs:4*bs], logits[4*bs:5*bs], logits[5*bs:]
                
                loss = (F.cross_entropy(logits_orig_0, targets) + F.cross_entropy(logits_orig_1, targets)) / 2.
                # print(loss)

                loss1 = consistency.consistency_loss([logits_orig_0, logits_orig_1],args.lbd1,loss='kl')
                loss2 = consistency.consistency_loss([logits_aug1_0, logits_aug1_1],args.lbd1,loss='kl')
                loss3 = consistency.consistency_loss([logits_aug2_0, logits_aug2_1],args.lbd1,loss='kl')

                loss += (loss1 + loss2 + loss3) / 3
                # print(loss)

                p_orig, p_augmix1, p_augmix2 = F.softmax(logits_orig_0, dim = -1)+F.softmax(logits_orig_1, dim = -1), F.softmax(logits_aug1_0, dim = -1)+F.softmax(logits_aug1_1, dim = -1), F.softmax(logits_aug2_0, dim = -1)+F.softmax(logits_aug2_1, dim = -1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_orig/2 + p_augmix1/2 + p_augmix2/2) / 3., 1e-7, 1).log()
                loss += args.lbd2 * (F.kl_div(p_mixture, p_orig/2, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix1/2, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix2/2, reduction='batchmean')) / 3.

            if js_loss and not new_loss:
                if i == 0:
                    test_img = torchvision.utils.make_grid(images[1], nrow = 16)
                    torchvision.utils.save_image(
                            test_img, "./test/fourier/test_3.png", nrow = 16
                        )
                    test_img = torchvision.utils.make_grid(images[0], nrow = 16)
                    torchvision.utils.save_image(
                            test_img, "./test/fourier/orig_3.png", nrow = 16
                        )
                bs = images[0].size(0)
                images_cat = torch.cat(images, dim = 0).to(device) # [3 * batch, 3, 32, 32]
                targets = targets.to(device)

                if args.scheme in ['half_ga','fourier_half_ga']:
                    index = np.random.choice(images_cat.shape[0],images_cat.shape[0]//2)
                    images_cat[index] = images_cat[index] + torch.randn_like(images_cat[index], device='cuda') * args.noise_sd
                elif args.scheme in ['ga','fourier_ga']:
                    images_cat = images_cat + torch.randn_like(images_cat, device='cuda') * args.noise_sd

                logits = model(images_cat)
                logits_orig, logits_augmix1, logits_augmix2 = logits[:bs], logits[bs:2*bs], logits[2*bs:]

                loss = F.cross_entropy(logits_orig, targets)

                p_orig, p_augmix1, p_augmix2 = F.softmax(logits_orig, dim = -1), F.softmax(logits_augmix1, dim = -1), F.softmax(logits_augmix2, dim = -1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_orig + p_augmix1 + p_augmix2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix2, reduction='batchmean')) / 3.

            elif not(js_loss):
                images, targets = images.to(device), targets.to(device)
                if args.scheme in ['half_ga','fourier_half_ga']:
                    index = np.random.choice(images.shape[0],images.shape[0]//2)
                    images[index] = images[index] + torch.randn_like(images[index], device='cuda') * args.noise_sd
                elif args.scheme in ['ga','fourier_ga']:
                    images = images + torch.randn_like(images, device='cuda') * args.noise_sd
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            losses.append(loss.item())
            if (i+1) % 10 == 0 or i+1 == len(train_loader):
                print("[%d/%d][%d/%d] Train Loss: %.4f | time : %.2fs"
                        %(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), time.time() - t))
                print("[%d/%d][%d/%d] Train Loss: %.4f | time : %.2fs"
                        %(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), time.time() - t), file=f, flush=True)
                t = time.time()

        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, args.outdir+"/%d.pt"%(epoch + 1))

            model.eval()
            with torch.no_grad():
                test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=args.batch,
                                    num_workers=args.workers, pin_memory=pin_memory)

                error, total = 0, 0
                print("Test on CIFAR-10")
                print("Test on CIFAR-10",file=f, flush= True)

                t = time.time()
                for i, (images, targets) in enumerate(test_loader):
                    images, targets = images.to(device), targets.to(device)
                    preds = torch.argmax(model(images), dim = -1)
                    error += (preds != targets).sum().item()
                    total += targets.size(0)

                print("Test error rate on CIFAR-10 : %.4f | time : %.2fs"%((error/total), time.time() - t))
                print("Test error rate on CIFAR-10 : %.4f | time : %.2fs"%((error/total), time.time() - t),file=f,flush=True)

                # evaluate on cifar10-c
                for corruption in CORRUPTIONS:
                    if args.dataset == 'cifar10':
                        test_data_c = cifar10_c.generate_all_examples(args.path,corruption)
                    elif args.dataset == 'cifar100':
                        test_data_c = cifar100_c.generate_all_examples(args.path,corruption)
                    print("Test on " + corruption)
                    print("Test on " + corruption,file=f,flush=True)
                    test_loader_c = torch.utils.data.DataLoader(test_data_c, shuffle=False, batch_size=args.batch,
                                    num_workers=args.workers, pin_memory=pin_memory)
                    error, total = 0, 0

                    t = time.time()
                    for i, (images, targets) in enumerate(test_loader_c):
                        images, targets = images.to(device), targets.to(device)
                        if images.shape[1] != 3:
                            images = images.permute(0,3,1,2)
                        preds = torch.argmax(model(images), dim = -1)
                        error += (preds != targets).sum().item()
                        total += targets.size(0)

                    print("Test error rate on CIFAR-10-C with " + corruption + " : %.4f | time : %.2fs"%(error/total, time.time() - t))
                    print("Test error rate on CIFAR-10-C with " + corruption + " : %.4f | time : %.2fs"%(error/total, time.time() - t),file=f,flush=True)
    
    f.close()

if __name__=="__main__":
    main()