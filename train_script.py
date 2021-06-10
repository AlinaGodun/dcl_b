# Importing all necessary libraries

# internal packages
import os

# external packages
import argparse
import sklearn

# util functions
from sklearn.cluster import KMeans

from util.util import *

# dataset functions
from dataset import load_util

# autoencoder
from models.simclr.simclr import SimCLR
from models.rotnet.rotnet import RotNet

from models.rotnet.IDEC import IDEC as RotNetIDEC
from models.simclr.IDEC import IDEC as SimClrIDEC


def train_model(model, batch_size, learning_rate, epochs, data, train, device, degree_of_space_distortion=0.1):
    print(f"Training {model.name} started...")
    model.to(device)

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

        model = model.fit(data_loader=trainloader, epochs=epochs, start_lr=learning_rate, device=device, model_path=pretrained_model_path, degree_of_space_distortion=degree_of_space_distortion)
        torch.save(model.state_dict(), pretrained_model_path)
    else:
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)

    return model


def perform_experiments(resnet_model='resnet18', epochs=20, learning_rates = [0.5, 1.0, 1.5]):
    for learning_rate in learning_rates:
        model = SimCLR(resnet_model=resnet_model)
        model.name = f'{model.name}_LR{learning_rate}_e{epochs}'
        train_model(model, batch_size, learning_rate, epochs, train, device)


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

device = detect_device()
print("Using device: ", device)

# specify learning params
batch_size = args.batch_size
learning_rate = args.lr
epochs = args.epochs

# training

train = True

# load datasets and create dataloaders
# data, testdata = load_util.load_cifar('./data', download=True, for_model='SimCLR')
# data_percent = args.data_percent
# data = load_util.load_custom_cifar('./data', download=False, data_percent=data_percent, for_model='RotNet')

clusterdata = load_util.load_custom_cifar('./data', download=False, data_percent=args.data_percent,
                                          train=True, transforms=False, for_model='SimCLR')
clusterloader = torch.utils.data.DataLoader(clusterdata,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

traindata = load_util.load_custom_cifar('./data', download=False, data_percent=args.data_percent,
                                        train=True, transforms=True, for_model='SimCLR')


# plot data
# plot_images(data[0:16])

# print('Data loaded...')
#
# # create model
#
# model = RotNet(num_classes=4)
# state_dict = torch.load(f'trained_models/pretrained_RotNet_features.pth', map_location='cpu')
# model.load_state_dict(state_dict)
# model.to(device)
#
# data = load_util.load_custom_cifar('./data', download=False, data_percent=data_percent, for_model='RotNet')
# trainloader = torch.utils.data.DataLoader(data,
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           drop_last=True)
#
# embedded_data, labels = model.encode_batchwise(trainloader, device=device, layer=['conv2'], flatten=True)
# n_clusters = len(set(labels))
# kmeans = KMeans(n_clusters=n_clusters)
# kmeans.fit(embedded_data)
# nmi = normalized_mutual_info_score(labels, kmeans.labels_)
# print(f"NMI: {nmi:.4f}")
#
# loss = torch.nn.CrossEntropyLoss()
# idec_simclr = IDEC(model, loss, kmeans.cluster_centers_, device)
# train_model(idec_simclr, batch_size, 0.001, epochs, data, train, device)

name = f'pretrained_SimCLR_r50_e1000.pth'
pretrained_model = load_model(name, device=device)
model = SimClrIDEC(pretrained_model, train_loader=clusterloader, device=device, n_clusters=10)
model.name = f'{model.name}_aug_e{epochs}'
print(f'training {model.name}')
print(f'base: {name}, epochs: {epochs}')
train_model(model, batch_size, learning_rate, epochs, traindata, train, device)

