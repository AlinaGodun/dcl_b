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
        self.rotation_num = 4
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}

        self.rotation_image_num = self.image_num * 4
        self.rotation_class_image_num = int(self.rotation_image_num / 4)

        # print('image num', self.image_num)
        # print('class image num', self.class_image_num)

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []

        for i in range(len(self.classes)):
            i_mask = targets == i
            self.data[i] = cifar.data[i_mask][:self.class_image_num]

            # print('class', i)
            # print('num of pics for class', len(self.data[i]))

            for d in self.data[i]:
                rotated_datas, rotated_labels = self.transforms(d)
                rotated_data_list += rotated_datas
                self.rotated_labels += rotated_labels

            # print('rotated labels num', len(self.rotated_labels))
            # print('rotated data pic num', len(rotated_data_list))

        rotated_labels_arrays = np.array(self.rotated_labels)
        rotated_data_list = np.array(rotated_data_list)

        print('final rotated labels', len(rotated_labels_arrays))
        print('final rotated imgs', len(rotated_data_list))

        print(rotated_labels_arrays)

        for i in range(len(self.transforms.rotate.keys())):
            i_mask = rotated_labels_arrays == i
            self.rotated_data[i] = rotated_data_list[i_mask]
            print(len(self.rotated_data[i]))


    def __len__(self):
        return self.rotation_image_num

    def __getitem__(self, idx):
        class_id = idx // self.rotation_class_image_num
        img_id = idx - class_id * self.rotation_class_image_num
        return self.transforms.one(self.rotated_data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.rotation_class_image_num
        return self.rotated_labels[class_id]
