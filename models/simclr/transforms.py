import torchvision
import numpy as np


class SimCLRTransforms:
    def __init__(self, with_original=False, grey=False):
        self.with_original = with_original
        self.to_tensor = torchvision.transforms.ToTensor()
        self.grey = grey
        self.to_pil = torchvision.transforms.ToPILImage()
        self.train_transform = torchvision.transforms.Compose([
             torchvision.transforms.RandomResizedCrop(32),
             torchvision.transforms.RandomHorizontalFlip(p=0.2),
             torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
             torchvision.transforms.RandomGrayscale(p=0.2)
             ])

    def __call__(self, x):
        xj = torchvision.transforms.ToTensor()(x) if isinstance(x, np.ndarray) else x
        t = [self.train_transform(xj), self.train_transform(xj)]
        if self.with_original:
            t.insert(0, xj)
        return tuple(t)

    def to_rgb(self, x):
      return self.to_tensor((self.to_pil(x).convert('RGB')))
