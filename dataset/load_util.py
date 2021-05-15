import torchvision
import torch

from dataset.ImageDataset import ImageDataset
from models.simclr.transforms import SimCLRTransforms
from models.simclr.custom_cifar import CustomCifar as SimCLRCustomCifar
from models.rotnet.custom_cifar import CustomCifar as RotNetCustomCifar

def load_mnist():
    # setup normalization function
    mnist_mean = 0.5
    mnist_std = 0.5
    normalize = torchvision.transforms.Normalize((mnist_mean,), (mnist_std,))
    # download the MNIST data set
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False)
    data = trainset.data
    # preprocess the data
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
    data = normalize(data)
    # Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    return data, labels


def load_cinic10(train_path):
    cinic_mean = 0.46
    cinic_std = 0.24
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    trainset = ImageDataset(root_dir = train_path, transforms=transforms, color=False)
    data = trainset.data.float()

    return data, trainset.labels


def load_cifar(train_path, download=False, for_model=None):
    model_transforms = {
        'SimCLR': SimCLRTransforms(),
        None: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    }
    cifar = torchvision.datasets.CIFAR10(root=train_path, train=True,
                                         download=download,
                                         transform=model_transforms[for_model])

    cifartest = torchvision.datasets.CIFAR10(root=train_path, train=False,
                                             download=False, transform=model_transforms[None])

    if for_model == 'SimCLR':
        return cifar, cifartest

    data = process_cifar_data(cifar.data)
    testdata = process_cifar_data(cifartest.data)
    return data, testdata


def load_custom_cifar(train_path, download=False, for_model=None, train=True, data_percent=1):
    if for_model == 'SimCLR' or for_model is None:
        return SimCLRCustomCifar(train_path, download=download, for_model=for_model, train=train, data_percent=data_percent)
    else:
        return RotNetCustomCifar(train_path, download=download, train=train, data_percent=data_percent)

def process_cifar_data(data):
    data = torch.Tensor(data)
    data = data/255
    data = data.permute(0, 3, 1, 2)
    return data
