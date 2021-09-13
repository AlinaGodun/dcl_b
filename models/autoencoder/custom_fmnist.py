import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset


class AEFMNIST(Dataset):
    def __init__(self, train_path='./data', download=False, data_percent=1.0, train=True, start='beginning'):
        """
        Custom wrapper for FMNIST dataset.

            Parameters:
                train_path (str): path to the dataset
                download (Boolean): if True, downloads dataset to the train_path, if False, looks for the dataset at
                train_path
                data_percent (float): percentage of data to be loaded
                train (Boolean): if True, loads train images; if False, loads test images

            Returns:
                AEFMNIST Dataset
        """
        fmnist = torchvision.datasets.FashionMNIST(root=train_path, train=train,
                                               download=download)

        targets = np.array(fmnist.targets)

        self.resize = torchvision.transforms.Resize(32)

        self.classes = fmnist.classes
        self.image_num = int(len(fmnist.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}

        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()

        for i in range(len(self.classes)):
            i_mask = targets == i
            t_data = []

            if start == 'beginning':
                data = fmnist.data.data[i_mask][:self.class_image_num]
            else:
                data = fmnist.data.data[i_mask][-self.class_image_num:]

            for d in data:
                t_data.append(self.to_tensor(self.to_pil(d).convert('RGB')))

            self.data[i] = self.resize(torch.stack(t_data)).float()

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


