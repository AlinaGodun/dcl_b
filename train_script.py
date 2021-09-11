# Importing all necessary libraries

# internal packages
import os

# external packages
import argparse
import sklearn

# util functions
from sklearn.cluster import KMeans

from models.autoencoder.IDEC import IDEC
from models.autoencoder.conv_ae import ConvAE
from models.autoencoder.custom_fmnist import AEFMNIST
from models.autoencoder.custom_stl10 import AESTL10
from models.simclr.custom_stl10 import SimCLRSTL10
from util.util import *

# dataset functions

# autoencoder
from models.simclr.simclr import SimCLR
from models.rotnet.rotnet import RotNet

from models.rotnet.DEC import DEC as RotNetIDEC
from models.simclr.IDEC import IDEC as SimClrIDEC
from models.rotnet.custom_stl10 import RotNetSTL10


def train_model(model, batch_size, learning_rate, epochs, data, train, device, degree_of_space_distortion=None):
    print(f"Training {model.name} started...")
    model = model.to(device)

    # paths to save/load models from
    base_path = "trained_models"
    pretrained_model_name = f"pretrained_{model.name}.pth"
    pretrained_model_path = os.path.join(base_path, pretrained_model_name)

    # training
    if train:
        # data_limit = int(len(data) * data_percent)
        # print(f"Number of train images: {data_limit}")
        print(f"Number of train images: {len(data)}")

        # trainloader = torch.utils.data.DataLoader(data[:data_limit],
        trainloader = torch.utils.data.DataLoader(data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

        if degree_of_space_distortion is None:
            model = model.fit(data_loader=trainloader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=pretrained_model_path)
        else:
            model = model.fit(data_loader=trainloader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=pretrained_model_path, degree_of_space_distortion=degree_of_space_distortion)
        torch.save(model.state_dict(), pretrained_model_path)
    else:
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)

    return model

def train_model2(model, batch_size, learning_rate, epochs, data, train, device, validation_dl, degree_of_space_distortion=None):
    print(f"Training {model.name} started...")
    model.to(device)

    # paths to save/load models from
    base_path = "trained_models"
    pretrained_model_name = f'{model.name}.pth'
    pretrained_model_path = os.path.join(base_path, pretrained_model_name)

    # training
    if train:
        trainloader = data

        if degree_of_space_distortion is None:
            model = model.fit(data_loader=trainloader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=pretrained_model_path, eval_data_loader=validation_dl)
        else:
            model = model.fit(data_loader=trainloader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=pretrained_model_path, degree_of_space_distortion=degree_of_space_distortion, eval_data_loader=validation_dl)
        torch.save(model.state_dict(), pretrained_model_path)
    else:
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)

    return model

parser = argparse.ArgumentParser(description='train_script')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--data_percent', type=float, default=0.4,
                    help='percent of data images to be used for training')
parser.add_argument('--resnet', type=str, default='resnet18',
                    help='default model used for SimCLR base')
args = parser.parse_args()

print("Versions:")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {np.__version__}",)
print(f"scikit-learn: {sklearn.__version__}")

# device = detect_device()
device = torch.device('cuda:1')
print("Using device: ", device)

# specify learning params
batch_size = args.batch_size
learning_rate = args.lr
epochs = args.epochs

# training

train = True

# clusterdata = load_util.load_custom_cifar('./data', download=False, data_percent=args.data_percent,
#                                           train=True, transforms=False, for_model='SimCLR')
# clusterloader = torch.utils.data.DataLoader(clusterdata,
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           drop_last=True)
#
# traindata = load_util.load_custom_cifar('./data', download=False, data_percent=args.data_percent,
#                                         train=True, transforms=True, for_model='SimCLR')
#
#
# name = f'pretrained_SimCLR_r50_e1000.pth'
# pretrained_model = load_model(name, device=device)
# model = SimClrIDEC(pretrained_model, train_loader=clusterloader, device=device, n_clusters=10)
# model.name = f'{model.name}_aug_e{epochs}'
# print(f'training {model.name}')
# print(f'base: {name}, epochs: {epochs}')
# train_model(model, batch_size, learning_rate, epochs, traindata, train, device)

# stl10 = SimCLRSTL10(download=False, data_percent=1.0, with_original=False)
# dataloader = torch.utils.data.DataLoader(stl10,
#                                          batch_size=128,
#                                          shuffle=True,
#                                          drop_last=True)

data = AEFMNIST('./data', data_percent=0.9)
validation_data = AEFMNIST('./data', data_percent=0.1, start='ending')

dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=128,
                                         shuffle=True,
                                         drop_last=True)

valloader = torch.utils.data.DataLoader(validation_data,
                                         batch_size=128,
                                         shuffle=True,
                                         drop_last=True)

for i in range(10):
    ae = ConvAE(n_channels=3, n_classes=3, embd_sz=64)
    ae.name = f'{ae.name}_STL10_{i}'
    # state_dict = torch.load(f'trained_models/pretrained_AE_{i}.pth', map_location=device)
    # ae.load_state_dict(state_dict)
    # ae = ae.to(device)

    # idec = IDEC(ae, dataloader, device)
    # idec.name = {f'{idec.name}_{i}'}
    train_model2(ae, 128, learning_rate, 300, dataloader, True, device, valloader)

# for i in range(5,10):
    # model = SimCLR()
    # model.name = f'{model.name}_STL10_{i}'
    # print(model.name)
    # train_model(model, batch_size, learning_rate, epochs, stl10, train, device)

    # model = load_model('pretrained_SimCLR_{i}.pth', device)
    # idec_model = SimClrIDEC(model, clusterloader, device)
    # idec_model.name = f'{idec_model.name}_{i}'
    # print(idec_model.name)
    # train_model(idec_model, batch_size, learning_rate, epochs, data, train, device)
