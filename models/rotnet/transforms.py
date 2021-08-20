import numpy as np
import torchvision
import torch


class RotNetTransforms:
    def __init__(self, grey=False):
        self.to_tensor_transform = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.resize = torchvision.transforms.Resize(32)
        self.rotate = {0: rotate_0, 1: rotate_90, 2: rotate_180, 3: rotate_270}
        self.grey = grey

    def __call__(self, x):
        if self.grey:
            rotated_xs = [r(x, self.grey) for _, r in self.rotate.items()]
            rotated_xs = [np.array(self.to_pil(x).convert('RGB')) for x in rotated_xs]
        else:
            rotated_xs = [r(x) for _, r in self.rotate.items()]
        rotated_labels = [label for label, _ in self.rotate.items()]
        return rotated_xs, rotated_labels

    def to_tensor(self, x):
        return self.to_tensor_transform(x)

    def resize(self, x):
        return self.resize(x)

    def get_tuple(self, x):
        rotated_xs, _ = self(x)
        return tuple(rotated_xs)


def rotate_0(x, grey=False):
    return x


def rotate_90(x, grey=False):
    if grey:
        return np.flipud(np.transpose(x, (1, 0))).copy()
    return np.flipud(np.transpose(x, (1, 0, 2))).copy()


def rotate_180(x, grey=False):
    return np.fliplr(np.flipud(x)).copy()


def rotate_270(x, grey=False):
    if grey:
        return np.transpose(np.flipud(x), (1, 0)).copy()
    return np.transpose(np.flipud(x), (1, 0, 2)).copy()
