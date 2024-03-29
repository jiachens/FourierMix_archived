'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-09 00:21:36
LastEditors: Jiachen Sun
LastEditTime: 2022-01-30 14:56:34
'''
import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.dncnn import DnCNN
from datasets import get_normalize_layer,get_input_center_layer
from torch.nn.functional import interpolate
from archs.resnet_18 import ResNet18
from archs.resnet_dubin import ResNet18_DuBIN

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110","cifar_resnet20_4","resnet_18","resnet_dubin"]

DENOISERS_ARCHITECTURES = ["cifar_dncnn", "cifar_dncnn_wide"
                        ]

IMAGENET_CLASSIFIERS = ["resnet50"]

def get_architecture(arch: str, dataset: str, normalize :bool = True,local_rank=None, device=None) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset in ["imagenet","imagenet-c","imagenet-c-bar"]:
        # model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda(}
        if local_rank != None:
            model = torch.nn.parallel.DistributedDataParallel(resnet50(pretrained=False).to(device), device_ids=[local_rank])
            model.to(device)
            # print("I AM HERE")
        else:
            model = resnet50(pretrained=False).cuda()
            # print("BUG HERE")
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet20_4":
        model = torch.nn.Sequential(resnet_cifar(depth=20, num_classes=4).cuda(),torch.nn.Softmax(dim=-1).cuda())
    elif arch == "cifar_resnet110" and dataset in ['cifar10','cifar10-c','cifar10-c-bar','cifar10-f']:
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "cifar_resnet110" and dataset in ['cifar100','cifar100-c','cifar100-c-bar','cifar100-f']:
        model = resnet_cifar(depth=110, num_classes=100).cuda()
    elif arch == "cifar_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        return model
    elif arch == "cifar_dncnn_wide":
        model = DnCNN(image_channels=3, depth=17, n_channels=128).cuda()
        return model
    elif arch == "resnet_18":
        model = ResNet18().cuda()
        return model
    elif arch == "resnet_dubin":
        model = ResNet18_DuBIN().cuda()
        return model
    if normalize:
        normalize_layer = get_input_center_layer(dataset)
        return torch.nn.Sequential(normalize_layer, model)
    else:
        return model
