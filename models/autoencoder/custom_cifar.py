import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset

from models.abstract_model.dataset import CustomCifar
from models.simclr.transforms import SimCLRTransforms


class AECIFAR(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, start='beginning'):
        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}
        data = cifar.data
        # data = data/255

        data = np.transpose(data, (0, 3, 1, 2))

        # self.mean = data.mean(axis=(0, 1, 2)) / 255
        # self.std = data.std(axis=(0, 1, 2)) / 255
        # normalize = torchvision.transforms.Normalize(self.mean, self.std)
        # t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                     torchvision.transforms.Normalize(self.mean, self.std)])
        # t = torchvision.transforms.ToTensor()

        for i in range(len(self.classes)):
            i_mask = targets == i
            if start == 'beginning':
                self.data[i] = torch.from_numpy(data[i_mask][:self.class_image_num]).float()
            else:
                self.data[i] = torch.from_numpy(data[i_mask][:-self.class_image_num]).float()
            # self.data[i] = normalize(self.data[i])

            # images = data[i_mask][:self.class_image_num]
            # t_images = []
            #
            # for img in images:
            #
            # self.data[i] = torch.stack(t_images)
            #     t_images.append(t(img))


    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.data[class_id][img_id], class_id

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]


