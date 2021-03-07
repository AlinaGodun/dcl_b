import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, root_dir, transforms, color):
        """
        Args:
            root_dir (string): Directory with all the images.
            transforms: transformation to be applied on images
            color: if images are color (3 channels) or not (1)
        """
        self.root = root_dir

        cdata = []
        clabels = []

        for idx, label in enumerate(os.listdir(self.root)):
            label_folder = os.path.join(self.root, label)

            if os.path.isdir(label_folder):
                for img in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img)

                    image = Image.open(img_path)
                    image = transforms(image)

                    # if color:
                    if image.shape[0] == 3:
                        image = image.reshape(-1, image.shape[0] * image.shape[1] * image.shape[2])
                        cdata.append(image)
                        clabels.append(idx)

                    # image = image.reshape(-1, image.shape[1] * image.shape[2])
                    # cdata.append(image)
                    # clabels.append(idx)

        self.data = torch.cat(cdata)
        self.labels = torch.Tensor(clabels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]