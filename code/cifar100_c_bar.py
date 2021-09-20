'''
Description: 
Autor: Jiachen Sun
Date: 2021-09-20 16:51:30
LastEditors: Jiachen Sun
LastEditTime: 2021-09-20 16:51:30
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset


_CIFAR_IMAGE_SIZE = (32, 32, 3)
_CIFAR_CLASSES = 100

_CORRUPTIONS_TO_FILENAMES = {
    'blue_noise_sample': 'blue_noise_sample.npy',
    'checkerboard_cutout': 'checkerboard_cutout.npy',
    'inverse_sparkles': 'inverse_sparkles.npy',
    'lines': 'lines.npy',
    'ripple': 'ripple.npy',
    'brownish_noise': 'brownish_noise.npy',
    'circular_motion_blur': 'circular_motion_blur.npy',
    'pinch_and_twirl': 'pinch_and_twirl.npy',
    'sparkles': 'sparkles.npy',
    'transverse_chromatic_abberation': 'transverse_chromatic_abberation.npy',
}

_CORRUPTIONS, _FILENAMES = zip(*sorted(_CORRUPTIONS_TO_FILENAMES.items()))
_DIRNAME = 'CIFAR-100-C-Bar'
_LABELS_FILENAME = 'labels.npy'


def generate_examples(data_dir,corruption,severity):
    corruption = corruption # _CORRUPTIONS
    severity = severity # (1,2,3,4,5)
    data_dir = os.path.join(data_dir,_DIRNAME)
    images_file = os.path.join(data_dir, _CORRUPTIONS_TO_FILENAMES[corruption])
    labels_file = os.path.join(data_dir, _LABELS_FILENAME)
    labels = np.load(labels_file)
    num_images = labels.shape[0] // 5
    labels = labels[:num_images]
    images = np.load(images_file)
    images = images[(severity - 1) * num_images:severity * num_images]
    # return zip(torch.Tensor(images), torch.Tensor(labels))
    dataset = CustomDataset(images,labels)
    # for (image, label) in zip(images, labels):
    #     dataset.append((torch.Tensor(image / 255), label))
    return dataset

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = torch.Tensor(self.data[idx] / 255)
        label = self.label[idx]
        return image, label