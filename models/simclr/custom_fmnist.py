import torchvision
import numpy as np
import torch
from models.abstract_model.dataset import CustomCifar
from models.simclr.transforms import SimCLRTransforms

class SimCLRFMNIST(CustomCifar):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, with_original=False, transforms=None):
        if transforms is None:
            transforms = train

        model_transforms = {
            True: SimCLRTransforms(with_original=with_original),
            False: torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        }

        fm = torchvision.datasets.FashionMNIST(root=train_path, train=train,
                                             download=download)

        targets = np.array(fm.targets)

        self.classes = fm.classes
        self.image_num = int(len(fm.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.transforms = model_transforms[transforms]
        self.data = {}

        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()

        for i in range(len(self.classes)):
            i_mask = targets == i
            t_data = []

            data = fm.data[i_mask][:self.class_image_num]
            for d in data:
              t_data.append(self.to_tensor((self.to_pil(d).convert('RGB'))))

            self.data[i] = np.transpose(torch.stack(t_data).numpy(), (0,3,2,1))

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
