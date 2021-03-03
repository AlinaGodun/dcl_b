import torchvision
import torch
from models.simclr.transforms import TransformsSimCLR
from main.util import *

cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False,
                                         transform=TransformsSimCLR())
d1, d2 = cifar.transform(cifar.data[0])
t = torch.cat((d1, d2), 0)
print(d1.shape, d2.shape)
print(cifar.data[0].shape)
plot_images(d1)
plot_images(d2)
