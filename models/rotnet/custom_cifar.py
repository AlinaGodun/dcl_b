import torchvision
import numpy as np
from torch.utils.data import Dataset
from models.rotnet.transforms import RotNetTransforms


class RotNetCIFAR(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True):
        self.transforms = RotNetTransforms()

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train, download=download)

        self.classes = cifar.classes
        self.rotation_num = 4
        self.rotation_image_num = int(len(cifar.data) * data_percent) * self.rotation_num
        self.rotation_class_image_num = int(self.rotation_image_num / self.rotation_num)

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []
        targets = np.array(cifar.targets)
        class_image_num = int(self.rotation_class_image_num / len(self.classes))

        for i in range(len(self.classes)):
            i_mask = targets == i
            data = cifar.data[i_mask][:class_image_num]

            for d in data:
                rotated_d, rotated_labels = self.transforms(d)
                imgs = [LocalImage(ro)]

                for rot_d, rot_label in zip(rotated_d, rotated_labels):
                    img = LocalImage(rot_d, )

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


class RotNetCIFARForPlot(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=False, with_original=True, transforms=True):
        self.transforms = RotNetTransforms()

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train,
                                             download=download)

        targets = np.array(cifar.targets)

        self.classes = cifar.classes
        self.image_num = int(len(cifar.data) * data_percent)
        self.class_image_num = int(self.image_num / len(self.classes))
        self.data = {}

        for i in range(len(self.classes)):
            i_mask = targets == i
            data_i = []
            original_data = cifar.data[i_mask][:self.class_image_num]

            for x in original_data:
                rot_d, rot_l = self.transforms(x)
                for rd, rl in zip(rot_d, rot_l):
                    data_i.append(LocalImage(self.transforms.to_tensor_transform(rd), rl))

            self.data[i] = data_i

        self.class_image_num *= 4
        self.image_num *= 4

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
    def __init__(self, image, original=0):
        self.image = image
        self.original = original


class RotNetCIFARForPlotAway(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True):
        self.transforms = RotNetTransforms()

        cifar = torchvision.datasets.CIFAR10(root=train_path, train=train, download=download)

        self.classes = cifar.classes
        self.rotation_num = 4
        self.rotation_image_num = int(len(cifar.data) * data_percent) * self.rotation_num
        self.rotation_class_image_num = int(self.rotation_image_num / self.rotation_num)

        self.rotated_data = {}
        self.rotated_labels = []

        rotated_data_list = []
        targets = np.array(cifar.targets)
        class_image_num = int(self.rotation_class_image_num / len(self.classes))

        for i in range(len(self.classes)):
            i_mask = targets == i
            data = cifar.data[i_mask][:class_image_num]

            for d in data:
                rotated_d, rotated_labels = self.transforms(d)
                # TODO: rotated_d is a list, add a label as the last point
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


class RotNetCIFARChanged(Dataset):
    def __init__(self, train_path, download=False, data_percent=0.4, train=True, transforms=None):
        if transforms is None:
            transforms = train
        model_transforms = {
            True: RotNetTransforms(),
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
        return self.transforms.get_tuple(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.class_image_num
        return self.classes[class_id]


