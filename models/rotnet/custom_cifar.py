import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.rotnet.transforms import RotNetTransforms


class CustomCifar(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True):
        self.transforms = RotNetTransforms()

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download,
                                             transform=self.transforms)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []

        for i in range(len(self.classes)):
            i_mask = targets == i
            self.data[i] = cifar.data[i_mask][:self.class_image_num]

            # TODO: add all the rotated data sorted by rotation class
            for d in self.data[i]:
                rotated_datas, rotated_labels = self.transforms(d)
                rotated_data_list += rotated_datas
                self.rotated_labels += self.rotated_labels

        for i in range(len(set(self.rotated_labels))):
            i_mask = self.rotated_data == i
            self.rotated_data[i] = rotated_data_list[i_mask][:self.class_image_num * 4]





    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.transforms(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]
