import numpy as np
import torch


class RotNetTransforms:
    def __init__(self):
        self.rotate = {0: rotate_0, 90: rotate_90, 180: rotate_180, 270: rotate_270}

    def __call__(self, x):
        rotated_xs = [r(x) for _, r in self.rotate.items()]
        rotated_labels = [label for label, _ in self.rotate.items()]
        return tuple(torch.stack(rotated_xs, dim=0), rotated_labels)


def rotate_0(x):
    return x


def rotate_90(x):
    return np.flipud(np.transpose(x, (1, 0, 2)))


def rotate_180(x):
    return np.fliplr(np.flipud(x))


def rotate_270(x):
    return np.transpose(np.flipud(x), (1, 0, 2))