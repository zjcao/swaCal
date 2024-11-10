"""
train: cinic10 or cifar10
coder: czjing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from engine.image_trainer import train_model
# from data.cifar10 import get_cifar10_dataloaders
from data.cinic10 import get_cinic10_dataloaders
from utils import plotAccLossCurves
# from models.resnet import get_resent50, get_wide_resent50_2, get_densenet121
from models.vgg import get_vgg16

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from engine.image_trainer import swa_train_model


# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# run normal.
def run(num_epochs=25):
    print("\n>>> start training ...\n")

    # -----------------
    # data
    cifar_dataloaders = get_cinic10_dataloaders(batch_size=64)

    # -----------------
    # model
    model_ft = get_vgg16()

    # -----------------
    # loss
    criterion = nn.CrossEntropyLoss()

    # -----------------
    # optimizer
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.002, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # -----------------
    # scheduler.
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=200, verbose=True)

    # -----------------
    # training
    model_ft, losses, accuracies = train_model(model_ft, cifar_dataloaders, criterion, optimizer_ft,
                                               exp_lr_scheduler, num_epochs=num_epochs, device=device)

    # -----------------
    # show losss
    plots = plotAccLossCurves(losses, accuracies)
    plots.acc_loss()




if __name__ == '__main__':

    # # normal.
    run()
