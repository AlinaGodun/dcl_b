import torchvision
import numpy as np
from models.abstract_model.dataset import CustomCifar
from models.simclr.transforms import SimCLRTransforms


class SimCLRCIFAR(CustomCifar):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, with_original=False, transforms=None):
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
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num
        return self.transforms(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
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
