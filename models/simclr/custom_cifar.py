import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.simclr.transforms import SimCLRTransforms
from models.rotnet.transforms import RotNetTransforms


class CustomCifar(Dataset):
    def __init__(self, train_path, download=False, for_model=None, data_percent=0.4, train=True):
        model_transforms = {
            'SimCLR': SimCLRTransforms(with_original=False),
            'Test': torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
            None: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        }

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download,
                                             transform=model_transforms[for_model])

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms['Test'] if not train else model_transforms[for_model]
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
