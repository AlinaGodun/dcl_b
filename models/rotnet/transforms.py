import numpy as np
import torchvision


class RotNetTransforms:
    def __init__(self, grey=False):
        """
        RotNet transformations. Available transformations: 0 degrees, 90 degrees, 180 degrees, 270 degrees

            Parameters:
                grey (Boolean): if True, converts greyscale to rgb to make the interaction with ConvNet based
                 models easier

            Returns:
                RotNet transformation object
        """
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.resize = torchvision.transforms.Resize(32)
        self.rotate = {0: self.rotate_0, 1: self.rotate_90, 2: self.rotate_180, 3: self.rotate_270}
        self.grey = grey

    def __call__(self, x):
        """
        Apply rotations to the input x.

            Parameters:
                x (tensor): input to be transformed

            Returns:
                (list of rotated xs, list of rotation labels)
        """
        rotated_xs = [r(x) for _, r in self.rotate.items()]
        rotated_labels = [label for label, _ in self.rotate.items()]

        if self.grey:
            rotated_xs = [np.array(self.to_pil(x).convert('RGB')) for x in rotated_xs]

        return rotated_xs, rotated_labels

    def get_tuple(self, x):
        rotated_xs, _ = self(x)
        return tuple(rotated_xs)

    def rotate_0(self, x):
        """
        Rotate x 0 degrees (just return the original input)

            Parameters:
                x (tensor): input to be transformed

            Returns:
                x turned 0 degrees
        """
        return x

    def rotate_90(self, x):
        """
        Rotate x 90 degrees

            Parameters:
                x (tensor): input to be transformed

            Returns:
                x turned 90 degrees
        """
        if self.grey:
            return np.flipud(np.transpose(x, (1, 0))).copy()
        return np.flipud(np.transpose(x, (1, 0, 2))).copy()

    def rotate_180(self, x):
        """
        Rotate x 180 degrees

            Parameters:
                x (tensor): input to be transformed

            Returns:
                x turned 180 degrees
        """
        return np.fliplr(np.flipud(x)).copy()

    def rotate_270(self, x):
        """
        Rotate x 270 degrees

            Parameters:
                x (tensor): input to be transformed

            Returns:
                x turned 90 degrees
        """
        if self.grey:
            return np.transpose(np.flipud(x), (1, 0)).copy()
        return np.transpose(np.flipud(x), (1, 0, 2)).copy()
