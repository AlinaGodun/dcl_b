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

# specify base paths

base_path = "material"
model_name = "autoencoder.pth"


print("Versions")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {np.__version__}",)
print(f"scikit-learn: {sklearn.__version__}")

data = dataset.load_util.load_cifar('./data')
plot_images(data[0:16])


