# from __future__ import print_function, division

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


# normal image model trainer.
def train_model(model, image_dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=None):
    since = time.time()

    if device is not None:
        model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Store losses and accuracies accross epochs (add by frank)
    losses, accuracies = dict(train=[], val=[]), dict(train=[], val=[])

    for epoch in range(num_epochs):
        print('Epoch {}/{} (lr: {:.3})'.format(epoch, num_epochs - 1, scheduler.get_last_lr()[0]))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # Iterate over data.
            for inputs, labels in image_dataloaders[phase]:
                if device is not None:
                    inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += labels.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.double() / running_total

            # # add by frank
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.cpu().numpy().item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    torch.save(model.state_dict(), ("./weights/model_epoch_{}_acc_{:.4f}.pth").format(epoch, epoch_acc))
            if phase == 'val' and epoch > num_epochs/2:
                model_wts = copy.deepcopy(model.state_dict())
                # save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # save best model
                    torch.save(best_model_wts, ("./weights/epoch_{}_model_acc_{:.4f}_best.pth").format(epoch, best_acc))
                elif epoch % 5 == 0 or epoch == num_epochs-1:
                    # save epoch model
                    torch.save(model_wts, ("./weights/epoch_{}_mdoel_acc_{:.4f}.pth").format(epoch, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save loss and acc
    np.save('./weights/acc_list.npy', accuracies)
    np.save('./weights/loss_list.npy', losses)

    return model, losses, accuracies


# swa image model trainer.
def swa_train_model(model, swa_model, swa_start, data_loaders, criterion,  optimizer, scheduler,
                    swa_scheduler, num_epochs=25, device=None):

    if device is not None:
        model = model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    # Store losses and accuracies accross epochs (add by frank)
    losses, accuracies = dict(train=[], val=[]), dict(train=[], val=[])

    for epoch in range(num_epochs):
        print('Epoch {}/{} (lr: {:.3})'.format(
            epoch, num_epochs - 1, scheduler.get_last_lr()[0] if epoch < swa_start else swa_scheduler.get_last_lr()[0]))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                if device is not None:
                    inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += labels.size(0)

                # only for debug...
                # if i >= 1:
                #     break


            if phase == 'train':
                if epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.double() / running_total

            # # add by frank
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.cpu().numpy().item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    torch.save(model.state_dict(), ("./weights/model_epoch_{}_acc_{:.4f}.pth").format(epoch, epoch_acc))
            if phase == 'val' and epoch > num_epochs/2:
                model_wts = copy.deepcopy(model.state_dict())
                # save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # save best model
                    torch.save(best_model_wts, ("./weights/epoch_{}_model_acc_{:.4f}_best.pth").format(epoch, best_acc))
                elif epoch % 5 == 0 or epoch == num_epochs-1:
                    # save epoch model
                    torch.save(model_wts, ("./weights/epoch_{}_mdoel_acc_{:.4f}.pth").format(epoch, epoch_acc))

        print()

    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn(data_loaders['train'], swa_model)
    swa_model_wts = copy.deepcopy(swa_model.state_dict())
    torch.save(swa_model_wts, "./weights/swa_mdoel.pth")
    print("swa_model saved.")
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    print("Best model loaded. ")
    model.load_state_dict(best_model_wts)

    # save loss and acc
    np.save('./weights/acc_list.npy', accuracies)
    np.save('./weights/loss_list.npy', losses)

    return model, swa_model, losses, accuracies


# main
if __name__ == '__main__':
    print("please run training()...")