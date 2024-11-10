import torch
import torch.nn as nn
import torchvision


def get_resent50(weights_path=None):
    # model
    model = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')

    # modify fc layer.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model


def get_wide_resent50_2(weights_path=None):
    # model
    model = torchvision.models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V2')

    # modify fc layer.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model

def get_densenet121(weights_path=None):
    # model
    model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")

    # modify fc layer.
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 10)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model


if __name__ == '__main__':

    # net
    net = get_resent50()
    print(net(torch.randn((2,3,32,32))).shape)
