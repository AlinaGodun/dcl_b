import torchvision


class SimCLRTransforms:
    def __init__(self, to_tensor=True):
        t = [torchvision.transforms.RandomResizedCrop(32),
             torchvision.transforms.RandomHorizontalFlip(p=0.2),
             torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
             torchvision.transforms.RandomGrayscale(p=0.2)]

        if to_tensor:
            t.append(torchvision.transforms.ToTensor())

        self.train_transform = torchvision.transforms.Compose(t)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
