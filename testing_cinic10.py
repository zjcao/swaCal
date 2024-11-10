
import torch
from engine.image_tester import cal_class_acc_ece
from data.cinic10 import get_cinic10_dataloaders
# from models.resnet import get_resent50, get_wide_resent50_2, get_densenet121
from models.vgg import get_vgg16
from torch.optim.swa_utils import AveragedModel

from utils import setting_seed
setting_seed(0)

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# testing normal.
def run():
    # ---------------
    # data
    cinic10_dataloaders = get_cinic10_dataloaders(batch_size=64,)

    # ---------------
    # vgg 16 model.
    weights_path = "/home/zjcao/work/open_code/swaCal/weights_pth/cinic10/sgd/epoch_24_mdoel_acc_0.7868.pth"
    model_ft = get_vgg16(weights_path=weights_path)

    # ---------------
    # get acc report.
    cal_class_acc_ece(model_ft, cinic10_dataloaders['test'], device=device)


# testing swa.
def run_swa():

    # ---------------
    # data
    cinic10_dataloaders = get_cinic10_dataloaders(batch_size=64, )

    # ---------------
    # vgg16 model.
    model = get_vgg16()

    # ---------------
    swa_model = AveragedModel(model)
    trained_weights_path = "/home/zjcao/work/open_code/swaCal/weights_pth/cinic10/swa/swa_mdoel.pth"
    swa_model.load_state_dict(torch.load(trained_weights_path, weights_only=True))

    # ---------------
    # get acc ece.
    cal_class_acc_ece(swa_model, cinic10_dataloaders['test'], device=device)


if __name__ == '__main__':
    # run()

    run_swa()