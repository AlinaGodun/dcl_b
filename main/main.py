# Importing all necessary libraries

# internal packages
import os
from collections import Counter, OrderedDict

# external packages
import torch
import torchvision
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
from torchsummary import summary

# util functions

from main.util import denormalize
from main.util import plot_images
from main.util import detect_device
from main.util import encode_batchwise
from main.util import decode_batchwise
from main.util import predict_batchwise
from main.util import evaluate_batchwise

# dataset functions
import dataset

# autoencoder
from models.autoencoder.conv_ae import ConvAE

print("Versions")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {np.__version__}",)
print(f"scikit-learn: {sklearn.__version__}")

device = detect_device()
print("Use device: ", device)

# specify learning params
batch_size = 256
learning_rate = 1e-3
epochs = 200

# create model
model = ConvAE(n_channels=3, n_classes=3)
model.to(device)

# load datasets and create dataloaders
data, testdata = dataset.load_util.load_cifar('./data')
plot_images(data[0:16])

data_percent = 0.4
data_limit = len(data) * data_percent
print(f"data_limit {data_limit}")


trainloader = torch.utils.data.DataLoader(data[:data_limit],
                                          batch_size=batch_size,
                                          # sample random mini-batches from the data
                                          shuffle=True,
                                          drop_last=False)

testloader = torch.utils.data.DataLoader(testdata,
                                         batch_size=batch_size,
                                         # Note that we deactivate the shuffling
                                         shuffle=False,
                                         drop_last=False)


# paths to save/load models from
model_name = "AE"
base_path = "trained_models"
pretrained_model_name = "pretrained_" + model_name
pretrained_model_path = os.path.join(base_path, pretrained_model_name)

# training
train = False

if train:
    model = model.fit(trainloader=trainloader, epochs=epochs, start_lr=learning_rate, end_lr=None, device=device)
    torch.save(model.state_dict(), pretrained_model_path)
else:
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)


