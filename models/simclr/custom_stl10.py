import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
from models.simclr.transforms import SimCLRTransforms


class SimCLRSTL10(Dataset):
    def __init__(self, train_path='./data', download=False, data_percent=1.0, train=True, with_original=True,
                 transforms=None):
        """
        Custom wrapper for STL10 dataset.

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
                SimCLRSTL10 Dataset
        """

        if transforms is None:
            transforms = train

        model_transforms = {
            True: SimCLRTransforms(with_original=with_original),
            False: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        }
        resize = torchvision.transforms.Resize(32)

        if train:
            stl10 = torchvision.datasets.STL10(train_path, download=download, split='unlabeled')
        else:
            stl10 = torchvision.datasets.STL10(train_path, download=download, split='test')

        self.class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        self.labels = stl10.labels
        self.classes = set(stl10.labels)
        self.image_num = int(len(stl10.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]

        data = stl10.data[:self.image_num, :]
        data = resize(torch.from_numpy(data)).numpy()
        self.data = np.transpose(data, (0, 2, 3, 1))

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
                idx (int): index of the image

            Returns:
                Image at index idx. If train or transforms is set to True, returns two augmented views of the image.
                If with_original is set to True, returns original image together with augmented views.
        """
        return self.transforms(self.data[idx]), self.labels[idx]

    def get_class(self, idx):
        """
        Returns class of the image at this index

            Parameters:
                idx (int): index of the image

            Returns:
                class of the image at this index
        """
        return self.labels[idx]
