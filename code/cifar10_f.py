'''
Description: 
Autor: Jiachen Sun
Date: 2021-08-18 01:28:17
LastEditors: Jiachen Sun
LastEditTime: 2021-10-10 15:13:05
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# _CORRUPTIONS_TO_FILENAMES = {
#     'abs_neg': 'abs_neg.npy',
#     'abs_pos': 'abs_neg.npy',
#     'angle': 'angle.npy'
# }

_DIRNAME = 'CIFAR-10-F'
_LABELS_FILENAME = 'label.npy'

def generate_examples(data_dir,corruption,severity):
    # corruption = corruption # _CORRUPTIONS
    # f_c,alpha = corruption
    severity = severity # (1,2,3,4,5)
    data_dir = os.path.join(data_dir,_DIRNAME)
    images_file = os.path.join(data_dir, 'fourier_' + corruption + '.npy')
    labels_file = os.path.join(data_dir, _LABELS_FILENAME)
    labels = np.load(labels_file)
    num_images = labels.shape[0] // 3
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