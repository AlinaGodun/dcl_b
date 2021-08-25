import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.simclr.transforms import SimCLRTransforms


class SimCLRCIFAR(Dataset):
    def __init__(self, train_path='./data', download=False, data_percent=1.0, train=True, with_original=False, transforms=None):
        """
        Custom wrapper for CIFAR dataset.

            Parameters:
                train_path (str): path to the dataset
                download (Boolean): if True, downloads dataset to the train_path, if False, looks for the dataset at
                train_path
                data_percent (float): percentage of data to be loaded
                train (Boolean): if True, loads train images; if False, loads test images
                with_original (Boolean): if True, SimCLRTransforms will load original image along with augmented views;
                if False, will load only augmented views
                transforms (Boolean): if True, transforms the images for training; if False, only changes images
                to tensors

            Returns:
                SimCLRCIFAR Dataset
        """
        if transforms is None:
            transforms = train

        model_transforms = {
            True: SimCLRTransforms(with_original=with_original),
            False: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        }

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]
        self.data = {}

        for i in range(len(self.classes)):
            i_mask = targets == i
            self.data[i] = cifar.data[i_mask][:self.class_image_num]

    def __len__(self):
        """
        Returns number of images in the dataset

            Returns:
                Number of images in the dataset
        """
        return self.image_num

    def __getitem__(self, idx):
        """
        Returns image at index idx. If train or transforms is set to True, returns two augmented views of the image.
        If with_original is set to True, returns original image together with augmented views.

            Parameters:
                idx (int): index

            Returns:
                Image at index idx. If train or transforms is set to True, returns two augmented views of the image.
                If with_original is set to True, returns original image together with augmented views.
        """
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.transforms(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
        """
        Returns class of the image at this index

            Parameters:
                idx (int): index of the image

            Returns:
                class of the image at this index
        """
        class_id = idx // self.class_image_num
        return self.classes[class_id]


class SimCLRCIFARForPlot(CustomCifar):
    def __init__(self, train_path, download=False, data_percent=0.4, train=False, with_original=True, transforms=True):
        if transforms is None:
            transforms = train

        model_transforms = {
            True: SimCLRTransforms(with_original=with_original),
            False: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        }

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]
        self.data = {}

        for i in range(len(self.classes)):
            i_mask = targets == i
            data_i = []
            original_data = cifar.data[i_mask][:self.class_image_num]

            for x in original_data:
                x, x_i, x_j = self.transforms(x)
                data_i.extend([LocalImage(x), LocalImage(x_i, 0), LocalImage(x_j, 0)])

            self.data[i] = data_i

        self.class_image_num *= 3
        self.image_num *= 3

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        d = self.data[class_id][img_id]
        return d.image, (class_id, d.original)

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]


class LocalImage:
    def __init__(self, image, original=1):
        self.image = image
        self.original = original
