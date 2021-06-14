'''
Description: 
Autor: Jiachen Sun
Date: 2021-06-10 13:47:49
LastEditors: Jiachen Sun
LastEditTime: 2021-06-10 16:42:58
'''
import os
import numpy as np
import torch

_DESCRIPTION = """\
Cifar10Corrupted is a dataset generated by adding 15 common corruptions + 4
extra corruptions to the test images in the Cifar10 dataset. This dataset wraps
the corrupted Cifar10 test images uploaded by the original authors.
"""

_CITATION = """\
@inproceedings{
  hendrycks2018benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HJz6tiCqYm},
}
"""

_CIFAR_IMAGE_SIZE = (32, 32, 3)
_CIFAR_CLASSES = 10

_CORRUPTIONS_TO_FILENAMES = {
    'gaussian_noise': 'gaussian_noise.npy',
    'shot_noise': 'shot_noise.npy',
    'impulse_noise': 'impulse_noise.npy',
    'defocus_blur': 'defocus_blur.npy',
    'frosted_glass_blur': 'glass_blur.npy',
    'motion_blur': 'motion_blur.npy',
    'zoom_blur': 'zoom_blur.npy',
    'snow': 'snow.npy',
    'frost': 'frost.npy',
    'fog': 'fog.npy',
    'brightness': 'brightness.npy',
    'contrast': 'contrast.npy',
    'elastic': 'elastic_transform.npy',
    'pixelate': 'pixelate.npy',
    'jpeg_compression': 'jpeg_compression.npy',
    'gaussian_blur': 'gaussian_blur.npy',
    'saturate': 'saturate.npy',
    'spatter': 'spatter.npy',
    'speckle_noise': 'speckle_noise.npy',
}
_CORRUPTIONS, _FILENAMES = zip(*sorted(_CORRUPTIONS_TO_FILENAMES.items()))
_DIRNAME = 'CIFAR-10-C'
_LABELS_FILENAME = 'labels.npy'

BENCHMARK_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
]

EXTRA_CORRUPTIONS = [
    'gaussian_blur',
    'saturate',
    'spatter',
    'speckle_noise',
]

def generate_examples(data_dir,corruption,severity):
    pass
    corruption = corruption # _CORRUPTIONS
    severity = severity # (1,2,3,4,5)

    images_file = os.path.join(data_dir, _CORRUPTIONS_TO_FILENAMES[corruption])
    labels_file = os.path.join(data_dir, _LABELS_FILENAME)
    labels = np.load(labels_file)
    num_images = labels.shape[0] // 5
    labels = labels[:num_images]
    images = np.load(images_file)
    images = images[(severity - 1) * num_images:severity * num_images]
    # return zip(torch.Tensor(images), torch.Tensor(labels))
    dataset = []
    for (image, label) in zip(images, labels):
        dataset.append((torch.Tensor(image / 255), label))
    return dataset