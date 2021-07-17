import torchvision
import numpy as np
import torch
from models.abstract_model.dataset import CustomCifar
from models.simclr.transforms import SimCLRTransforms


class SimCLRSTL10(CustomCifar):
    def __init__(self, train_path='./data', download=False, data_percent=0.4, train=True, with_original=True, transforms=None):
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

        self.labels = stl10.labels
        self.classes = set(stl10.labels)
        self.image_num = int(len(stl10.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]
        self.data = {}

        data = stl10.data[:self.image_num, :]
        data = resize(torch.from_numpy(data)).numpy()
        self.data = np.transpose(data, (0, 2, 3, 1))

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.transforms(self.data[idx]), self.labels[idx]

    def get_class(self, idx):
        return -1
