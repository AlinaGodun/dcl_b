import torchvision
import torch

from dataset.ImageDataset import ImageDataset


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


def load_cifar(train_path):
    cifar = torchvision.datasets.CIFAR10(root=train_path, train=True,
                                         download=False,
                                         transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]))
    data = cifar.data
    data = torch.Tensor(data)
    data = data/255
    data = data.permute(0, 3, 1, 2)
    # data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    return data
