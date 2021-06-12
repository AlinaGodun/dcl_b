import numpy as np
import torchvision


class RotNetTransforms:
    def __init__(self):
        self.to_tensor_transform = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(32)
        self.rotate = {0: rotate_0, 1: rotate_90, 2: rotate_180, 3: rotate_270}

    def __call__(self, x):
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


def rotate_0(x):
    return x


def rotate_90(x):
    return np.flipud(np.transpose(x, (1, 0, 2))).copy()


def rotate_180(x):
    return np.fliplr(np.flipud(x)).copy()


def rotate_270(x):
    return np.transpose(np.flipud(x), (1, 0, 2)).copy()
