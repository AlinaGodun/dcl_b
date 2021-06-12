import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.rotnet.transforms import RotNetTransforms


class RotNetSTL10(Dataset):
    def __init__(self, train_path='./data', download=True, data_percent=0.4, train=True):
        self.transforms = RotNetTransforms()

        stl10 = torchvision.datasets.STL10(train_path, download=download, split='unlabeled')

        self.rotation_num = 4
        self.rotation_image_num = int(len(stl10.data) * data_percent) * self.rotation_num
        self.rotation_class_image_num = int(len(stl10.data) * data_percent)

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []

        data = np.transpose(stl10.data, (0, 2, 3, 1))
        for d in data:
            rotated_d, rotated_labels = self.transforms(d)
            rotated_data_list += rotated_d
            self.rotated_labels += rotated_labels

        self.rotated_labels = np.array(self.rotated_labels)
        rotated_data_list = np.array(rotated_data_list)

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