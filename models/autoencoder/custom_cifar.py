import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset


class AECIFAR(Dataset):
    def __init__(self, train_path='./data', download=False, data_percent=1.0, train=True, start='beginning', transforms=False):
        """
        Custom wrapper for CIFAR dataset.

            Parameters:
                train_path (str): path to the dataset
                download (Boolean): if True, downloads dataset to the train_path, if False, looks for the dataset at
                train_path
                data_percent (float): percentage of data to be loaded
                train (Boolean): if True, loads train images; if False, loads test images

            Returns:
                AECIFAR Dataset
        """
        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}
        data = cifar.data

        data = np.transpose(data, (0, 3, 1, 2))

        for i in range(len(self.classes)):
            i_mask = targets == i
            if start == 'beginning':
                self.data[i] = torch.from_numpy(data[i_mask][:self.class_image_num]).float()
            else:
                self.data[i] = torch.from_numpy(data[i_mask][-self.class_image_num:]).float()

    def __len__(self):
        """
        Returns number of images in the dataset

            Returns:
                Number of images in the dataset
        """
        return self.image_num

    def __getitem__(self, idx):
        """
        Returns image at index idx.

            Parameters:
                idx (int): index

            Returns:
                Image at index idx.
        """
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.data[class_id][img_id], class_id

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


