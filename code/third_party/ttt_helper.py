'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-13 18:06:28
LastEditors: Jiachen Sun
LastEditTime: 2021-07-15 17:28:56
'''
import numpy as np
import torch
import torch.nn as nn
from architectures import *
import torchvision.transforms as transforms

te_transforms = transforms.Compose([transforms.ToTensor()])

def build_model(args):
    from archs.cifar_resnet import resnet as resnet_cifar
    from archs.cifar_ssl import ExtractorHead
    print('Building model...')

    net = get_architecture(args.arch, args.dataset,args.no_normalize)

    if args.shared == 'layer3' or args.shared is None:
        from archs.cifar_ssl import extractor_from_layer3
        ext = extractor_from_layer3(net)
        head = nn.Linear(64, 4)
    elif args.shared == 'layer2':
        from archs.cifar_ssl import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net, 4)
    ssh = ExtractorHead(ext, head).cuda()

    # if len(args.gpu) > 1:
    net = torch.nn.DataParallel(net)
    ssh = torch.nn.DataParallel(ssh)
    
    return net, ext, head, ssh

def rotate_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
		batch = batch.repeat((4,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels

def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		if label == 1:
			img = tensor_rot_90(img)
		elif label == 2:
			img = tensor_rot_180(img)
		elif label == 3:
			img = tensor_rot_270(img)
		images.append(img.unsqueeze(0))
	return torch.cat(images)