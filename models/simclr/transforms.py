import torchvision


class SimCLRTransforms:
    def __init__(self):
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=96),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(degrees=90),
                torchvision.transforms.ColorJitter(0.5, 0.5, 0.5),
                torchvision.transforms.RandomGrayscale(),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
