# Importing all necessary libraries

# internal packages
import os
from collections import Counter, OrderedDict

# external packages
import torch
import torchvision
import numpy as np
import sklearn
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    print("fuck you")
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
from torchsummary import summary

# multithreading
from multiprocessing import Process

# util functions
from main.util import denormalize
from main.util import plot_images
from main.util import detect_device
from main.util import encode_batchwise
from main.util import decode_batchwise
from main.util import predict_batchwise
from main.util import evaluate_batchwise

# dataset functions
from dataset import load_util

# autoencoder
from models.autoencoder.conv_ae import ConvAE


def train_model(model, batch_size, learning_rate, epochs, data, data_percent, train, device):
    print(f"Training {model.name} started...")
    model.to(device)

    # paths to save/load models from
    base_path = "trained_models"
    pretrained_model_name = "pretrained_" + model.name
    pretrained_model_path = os.path.join(base_path, pretrained_model_name)

    # training
    if train:
        data_limit = int(len(data) * data_percent)
        print(f"Number of train images: {data_limit}")

        trainloader = torch.utils.data.DataLoader(data[:data_limit],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=False)

        model = model.fit(trainloader=trainloader, epochs=epochs, start_lr=learning_rate, device=device, model_path=pretrained_model_path)
        torch.save(model.state_dict(), pretrained_model_path)
    else:
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':
    print("New concurrent")
    print("Versions:")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"numpy: {np.__version__}",)
    print(f"scikit-learn: {sklearn.__version__}")

    device = detect_device()
    print("Using device: ", device)

    # specify learning params
    batch_size = 256
    learning_rate = 1e-3
    epochs = 5

    # training
    train = True

    # load datasets and create dataloaders
    data, testdata = load_util.load_cifar('./data')
    data_percent = 0.4

    # plot data
    # plot_images(data[0:16])

    print('Data loaded...')

    # create model
    # model = ConvAE(n_channels=3, n_classes=3)

    # model = train_model(model, batch_size, learning_rate, epochs, data, data_percent, train, device)

    args_list = []

    for i in range(5):
        model = ConvAE(n_channels=3, n_classes=3, name=f"ConvAE{i}")
        args = (model, batch_size, learning_rate, epochs, data, data_percent, train, device)
        args_list.append(args)

    with Pool(processes=5) as pool:
        models = pool.starmap(train_model, args_list)

    for m in models:
        print(m.name)

    # processes = []

    # for i in range(4):
    #     model = ConvAE(n_channels=3, n_classes=3, name=f"ConvAE{i}")
    #     args = (model, batch_size, learning_rate, epochs, data, data_percent, train, device)
    #     p = Process(target=train_model, args=args)
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # print('ready')

