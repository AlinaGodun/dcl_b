import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.rotnet.transforms import RotNetTransforms


class RotNetFashionMNIST(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, start='beginning'):
        """
        Custom wrapper for Fashion MNIST dataset.

            Parameters:
                train_path (str): path to the dataset
                download (Boolean): if True, downloads dataset to the train_path, if False, looks for the dataset at
                train_path
                data_percent (float): percentage of data to be loaded
                train (Boolean): if True, loads train images; if False, loads test images

            Returns:
                Fashion MNIST Dataset
        """
        self.transforms = RotNetTransforms(grey=True)

        fm = torchvision.datasets.FashionMNIST(root=train_path, train=train, download=download)

        self.classes = fm.classes
        self.rotation_num = 4
        self.rotation_image_num = int(len(fm.data) * data_percent) * self.rotation_num
        self.rotation_class_image_num = int(self.rotation_image_num / self.rotation_num)

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []
        targets = np.array(fm.targets)
        class_image_num = int(self.rotation_class_image_num / len(self.classes))

        for i in range(len(self.classes)):
            i_mask = targets == i
            if start == 'beginning':
                data = fm.data[i_mask][:class_image_num]
            else:
                data = fm.data[i_mask][:-class_image_num]

            for d in data:
                rotated_d, rotated_labels = self.transforms(d)
                rotated_data_list += rotated_d
                self.rotated_labels += rotated_labels

        self.rotated_labels = np.array(self.rotated_labels)

        print(rotated_data_list[0].shape)
        print(type(rotated_data_list[0]))
        rotated_data_list = np.array(rotated_data_list)

        for i in range(len(self.transforms.rotate.keys())):
            i_mask = self.rotated_labels == i
            self.rotated_data[i] = rotated_data_list[i_mask]

    def __len__(self):
        """
        Returns number of images in the dataset

            Returns:
                Number of images in the dataset
        """
        return self.rotation_image_num

    def __getitem__(self, idx):
        """
        Returns image at index idx.

            Parameters:
                idx (int): index

            Returns:
                Image at index idx.
        """
        class_id = idx // self.rotation_class_image_num
        img_id = idx - class_id * self.rotation_class_image_num
        return self.transforms.to_tensor(self.rotated_data[class_id][img_id]), class_id

    def get_class(self, idx):
        """
        Returns class of the image at this index

            Parameters:
                idx (int): index of the image

            Returns:
                class of the image at this index
        """
        class_id = idx // self.rotation_class_image_num
        return self.rotated_labels[class_id]

