import numpy as np
import torch
import torchvision


class RotNetTransforms:
    def __init__(self):
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.rotate = {0: rotate_0, 90: rotate_90, 180: rotate_180, 270: rotate_270}

    def __call__(self, x):
        rotated_xs = [self.train_transform(r(x)) for _, r in self.rotate.items()]
        rotated_labels = [label for label, _ in self.rotate.items()]
        return rotated_xs, rotated_labels


def rotate_0(x):
    return x


def rotate_90(x):
    return np.flipud(np.transpose(x, (1, 0, 2))).copy()


def rotate_180(x):
    return np.fliplr(np.flipud(x)).copy()


def rotate_270(x):
    return np.transpose(np.flipud(x), (1, 0, 2)).copy()
