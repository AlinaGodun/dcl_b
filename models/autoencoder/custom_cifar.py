import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset

from models.abstract_model.dataset import CustomCifar
from models.simclr.transforms import SimCLRTransforms


class AECIFAR(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True):
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
            self.data[i] = torch.from_numpy(data[i_mask][:self.class_image_num]).float()

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.data[class_id][img_id], class_id

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]
