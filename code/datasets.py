from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import cifar10_c
import cifar10_c_bar
import cifar10_f
import cifar100_f
import cifar100_c
import cifar100_c_bar
import transformation

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
os.environ["IMAGENET_DIR"] = '/usr/workspace/safeml/data/james-imagenet'
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "imagenet-c","imagenet-c-bar","cifar10", "cifar10-c", "cifar10-c-bar","cifar10-f","cifar100-f","cifar100","cifar100-c","cifar100-c-bar"]


def get_dataset(dataset: str, split: str, data_dir=None,corruption=None,severity=None,scheme = None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, scheme)
    if dataset == "imagenet-c":
        return _imagenet_c(corruption, severity)
    if dataset == "imagenet-c-bar":
        return _imagenet_c_bar(corruption, severity)
    elif dataset == "cifar10":
        return _cifar10(split, scheme, severity)
    elif dataset == "cifar100":
        return _cifar100(split, scheme, severity)
    elif dataset == "cifar100-c":
        return _cifar100_c(data_dir,corruption, severity)
    elif dataset == "cifar100-c-bar":
        return _cifar100_c_bar(data_dir,corruption, severity)
    elif dataset == "cifar10-c":
        return _cifar10_c(data_dir,corruption,severity)
    elif dataset == "cifar10-c-bar":
        return _cifar10_c_bar(data_dir,corruption,severity)
    elif dataset == "cifar10-f":
        return _cifar10_f(data_dir,corruption,severity)
    elif dataset == "cifar100-f":
        return _cifar100_f(data_dir,corruption,severity)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset in ["imagenet","imagenet-c","imagenet-c-bar"]:
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar10-c":
        return 10
    elif dataset == "cifar10-c-bar":
        return 10
    elif dataset == "cifar10-f":
        return 10
    elif dataset in ["cifar100","cifar100-c","cifar100-c-bar","cifar100-f"]:
        return 100


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar10-c":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar10-c-bar":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar10-f":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar100":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str, scheme, severity: int) -> Dataset:
    if split == "train":
        if scheme in ['contrast_ga']:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transformation.Contrast(level=5,maxval=1.8),
                transforms.ToTensor()
            ]))
        elif scheme in ['contrast_2_ga']:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transformation.Contrast_2(severity=severity)
            ]))
        elif scheme in ['fog_ga']:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transformation.Fog(severity=severity)
            ]))
        elif scheme == "autocontrast":
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
            ]))
        elif scheme in ["augmix","augmix_half_ga","augmix_ga","expert_half_ga","expert_ga"]:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]))
        elif scheme in ["fourier_half_ga"]:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        elif scheme in ["auto_half_ga","pg_half_ga","half_ga_jsd"]:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=None)
        elif scheme in ["c_half_ga"]:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
        else:
            return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
    elif split == "test":
        if scheme in ["expert_half_ga","expert_ga"]:
            return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=None)
        else:
            return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _cifar100(split: str, scheme, severity: int) -> Dataset:
    if split == "train":
        if scheme in ["fourier_half_ga"]:
            return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        elif scheme in ["augmix","augmix_half_ga","augmix_ga","expert_half_ga","expert_ga"]:
            return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]))
        elif scheme in ["auto_half_ga"]:
            return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=None)
        else:
            return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
    elif split == "test":
        return datasets.CIFAR100("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _cifar10_c(data_dir: str, corruption: str, severity: int) -> Dataset:
    return cifar10_c.generate_examples(data_dir,corruption,severity)

def _cifar100_c(data_dir: str, corruption: str, severity: int) -> Dataset:
    return cifar100_c.generate_examples(data_dir,corruption,severity)

def _cifar10_c_bar(data_dir: str, corruption: str, severity: int) -> Dataset:
    return cifar10_c_bar.generate_examples(data_dir,corruption,severity)

def _cifar100_c_bar(data_dir: str, corruption: str, severity: int) -> Dataset:
    return cifar100_c_bar.generate_examples(data_dir,corruption,severity)

def _cifar10_f(data_dir: str, corruption, severity: int) -> Dataset:
    return cifar10_f.generate_examples(data_dir,corruption,severity)

def _cifar100_f(data_dir: str, corruption, severity: int) -> Dataset:
    return cifar100_f.generate_examples(data_dir,corruption,severity)


def _imagenet(split: str, scheme: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        if scheme in ['augmix_half_ga']:
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip()
            ])
        elif scheme in ['fourier_half_ga']:
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                 transforms.ToTensor()
            ])
        elif scheme in ['half_ga_jsd']:
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224)
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _imagenet_c(corruption: str, severity: int) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")
    
    if corruption == 'frosted_glass_blur':
        corruption = 'glass_blur'
    if corruption == 'elastic':
        corruption = 'elastic_transform'

    dir = "/usr/workspace/safeml/data/imagenet-c/" + corruption + '/' + str(severity)
    # subdir = os.path.join(dir, "val")
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return datasets.ImageFolder(dir, transform)

def _imagenet_c_bar(corruption: str, severity: int) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    
    root = "/usr/workspace/safeml/data/img-c-bar/ImageNet-C-Bar/" + corruption
    g = os.listdir(root)  
    # g = [float(x) for x in g]
    # g.sort()

    dir = "/usr/workspace/safeml/data/img-c-bar/ImageNet-C-Bar/" + corruption + '/' + str(g[severity-1])
    # subdir = os.path.join(dir, "val")
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return datasets.ImageFolder(dir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.means = torch.tensor(means).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means