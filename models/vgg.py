import torch
import torch.nn as nn
import torchvision


def get_vgg16(weights_path=None):
    # model
    model = torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")

    # modify fc layer.
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True))

    return model


if __name__ == '__main__':

    # vgg16
    weight_path = "/home/zjcao/work/open_code/swaCal/weights_pth/cinic10/sgd/epoch_24_mdoel_acc_0.7868.pth"
    vgg16 = get_vgg16(weights_path=weight_path)
    print(vgg16(torch.randn((2,3,32,32))).shape)
