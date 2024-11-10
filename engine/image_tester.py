"""
engine: image tester.
coder: czjing
"""

import time
import os
import copy
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# get class acc report
def cal_class_acc_ece(model, dataloader, device=None):

    if device is not None:
        model = model.to(device)

    # prepare to count predictions for each class
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in label_names}
    total_pred = {classname: 0 for classname in label_names}

    y_preds = []        # 预测标签 y_preds:[0,1,2,3,4,5,6,7,8,9,10]
    y_preds_probs = []  # 预测概率 y_hat pros: [[0.0,...1.0], [0.0,...1.0]...]
    y_trues = []        # 参考标签 y_labels: [0,1,2,3,4,5,6,7,8,9,10]

    # again no gradients needed
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            if device is not None:
                images, labels = images.to(device), labels.to(device)

            # logits
            logits = model(images)                   # logits: [-oo,+oo]

            # preds
            _, preds = torch.max(logits, 1)          # preds: tensor([3, 8, 8, 0, 4, 6, 1, 2, 5, 9])

            # probs
            probs = torch.nn.Softmax(dim=1)(logits)  # probs: tensor([[3.4013e-04, 1.6955e-04, 7.3739e-03, 6.5632e-01, ...]])

            # total acc
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # calculate class acc
            for label, pred, prob in zip(labels, preds, probs):
                # class acc
                if label == pred:
                    correct_pred[label_names[label]] += 1
                total_pred[label_names[label]] += 1

                # outputs
                y_preds.append(pred.item())
                # y_preds_probs.append(prob[pred].item())
                y_preds_probs.append(prob.tolist())
                y_trues.append(label.item())

    # calculate accuracy for each class
    # using sklearn metrics
    from sklearn.metrics import classification_report
    acc_report = classification_report(y_trues, y_preds, digits=4)
    print(acc_report)

    # calculate expected calibration error
    from torchmetrics.classification import MulticlassCalibrationError
    norms = ['l1', 'l2', 'max']
    ece = []
    for norm in norms:
        metric = MulticlassCalibrationError(num_classes=10, n_bins=10, norm=norm)
        ece.append(metric(torch.tensor(y_preds_probs), torch.tensor(y_trues)).numpy())
    print("ECE: {} | MCE: {} | RMSCE: {}".format(ece[0], ece[1], ece[2]))
    print("Avg. Confi. {}".format(np.mean(np.max(y_preds_probs,1))))

    # return acc_report, ece



# main()
if __name__ == '__main__':
    print("please run other script.")


