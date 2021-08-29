import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.simclr.transforms import SimCLRTransforms

class SimCLRFMNIST(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, with_original=False, transforms=None):
        if transforms is None:
            transforms = train

        model_transforms = {
            True: SimCLRTransforms(with_original=with_original),
            False: torchvision.transforms.Compose([])
        }

        fm = torchvision.datasets.FashionMNIST(root=train_path, train=train,
                                               download=download)

        targets = np.array(fm.targets)

        self.classes = fm.classes
        self.image_num = int(len(fm.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]
        self.data = {}

        for i in range(len(self.classes)):
            i_mask = targets == i
        t_data = []

        data = fm.data[i_mask][:self.class_image_num]
        for d in data:
            t_data.append(model_transforms[True].to_rgb(d))

        self.data[i] = torch.stack(t_data)
        # self.data[i] = fm.data[i_mask][:self.class_image_num]

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.class_image_num
        img_id = idx - class_id * self.class_image_num

        return self.transforms(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]
