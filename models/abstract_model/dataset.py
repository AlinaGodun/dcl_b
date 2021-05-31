from torch.utils.data import Dataset


class CustomCifar(Dataset):
    def __init__(self, classes, image_num, transforms, data):
        self.classes = classes
        self.image_num = image_num
        self.image_num_per_class = int(self.image_num / len(self.classes))
        self.transforms = transforms
        self.data = data

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        class_id = idx // self.image_num_per_class
        img_id = idx - class_id * self.image_num_per_class
        return self.transforms(self.data[class_id][img_id]), class_id

    def get_class(self, idx):
        class_id = idx // self.image_num_per_class
        return self.classes[class_id]
