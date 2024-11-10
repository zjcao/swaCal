"""
CINIC-10 dataset
"""

import time
import os
import copy
import logging
from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from utils import img_show


def get_cinic10_datasets():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ]),
    }

    data_dir = '/data/zjcao/data/CINIC/'
    cinic10_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        )
        for x in ['train', 'val', 'test']}

    return cinic10_datasets


def get_cinic10_dataloaders(batch_size):

    cinic10_datasets = get_cinic10_datasets()

    cinic10_dataloaders = {
        x: torch.utils.data.DataLoader(
            cinic10_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val', 'test']
    }

    return cinic10_dataloaders



if __name__ == '__main__':

    # get data loader
    cinic10_dataloaders = get_cinic10_dataloaders(batch_size=32)

    # data info.
    cinic10_datasets = get_cinic10_datasets()
    cinic10_dataset_sizes = {x: len(cinic10_datasets[x]) for x in ['train', 'val', 'test']}
    print("Image dataset info.: Train set: {} | Val set: {} | Test set: {}"
        .format(cinic10_dataset_sizes['train'],
                cinic10_dataset_sizes['val'],
                cinic10_dataset_sizes['test']))

    # testing dataloaders
    _X, _y = next(iter(cinic10_dataloaders['train']))
    print("Inputs shape: {} | labels(0:10): {}".format(_X.numpy().shape, _y.numpy()[0:10]))

    # make a grid from batch
    out = torchvision.utils.make_grid(_X[0:8])
    label_names =get_cinic10_datasets()['train'].classes
    img_show(out, title=[label_names[x] for x in _y[0:8]])