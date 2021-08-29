import torchvision
import numpy as np


class SimCLRTransforms:
    def __init__(self, with_original=False):
        """
        SimCLR transformations. Consists of random resized crop, random horizontal flip, color jitter and random
            grayscale.

            Parameters:
                with_original (Boolean): if True, returns original image x together with its augmented views when
                transforming the image

            Returns:
                SimCLR transformation object
        """
        self.with_original = with_original
        self.train_transform = torchvision.transforms.Compose([
             torchvision.transforms.RandomResizedCrop(32),
             torchvision.transforms.RandomHorizontalFlip(p=0.2),
             torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
             torchvision.transforms.RandomGrayscale(p=0.2)
             ])

    def __call__(self, x):
        """
        Apply transformation to the input x.

            Parameters:
                x (tensor): input to be transformed

            Returns:
                (augmented x 1, augmented x 2, (optional) original x) - original x is returned if with_original = True
                when SimCLRTransforms was created
        """
        xj = torchvision.transforms.ToTensor()(x) if isinstance(x, np.ndarray) else x
        t = [self.train_transform(xj), self.train_transform(xj)]
        if self.with_original:
            t.insert(0, xj)
        return tuple(t)

    def to_rgb(self, x):
        x = self.to_tensor(x) if isinstance(x, np.ndarray) else x
        return self.to_tensor(self.to_pil(x).convert('RGB'))