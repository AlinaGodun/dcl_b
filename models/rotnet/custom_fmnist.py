import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
from models.rotnet.transforms import RotNetTransforms


class RotNetFashionMNIST(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True):
        self.transforms = RotNetTransforms(grey=True)

        fm = torchvision.datasets.FashionMNIST(root=train_path, train=train, download=download)


        print('1')
        self.classes = fm.classes
        self.rotation_num = 4
        self.rotation_image_num = int(len(fm.data) * data_percent) * self.rotation_num
        self.rotation_class_image_num = int(self.rotation_image_num / self.rotation_num)

        print('2')
        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []
        targets = np.array(fm.targets)
        class_image_num = int(self.rotation_class_image_num / len(self.classes))

        print('3')
        for i in range(len(self.classes)):
            i_mask = targets == i
            data = fm.data[i_mask][:class_image_num]

            for d in data:
                rotated_d, rotated_labels = self.transforms(d)
                rotated_data_list += rotated_d
                self.rotated_labels += rotated_labels

        print('4')
        self.rotated_labels = np.array(self.rotated_labels)

        print(rotated_data_list[0].shape)
        print(type(rotated_data_list[0]))
        rotated_data_list = np.array(rotated_data_list)

        print('5')
        for i in range(len(self.transforms.rotate.keys())):
            i_mask = self.rotated_labels == i
            self.rotated_data[i] = rotated_data_list[i_mask]

    def __len__(self):
        return self.rotation_image_num

    def __getitem__(self, idx):
        class_id = idx // self.rotation_class_image_num
        img_id = idx - class_id * self.rotation_class_image_num
        return self.transforms.to_tensor_transform(self.rotated_data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.rotation_class_image_num
        return self.rotated_labels[class_id]

