from abc import abstractmethod
import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, name, loss):
        super().__init__()
        self.name = name
        self.loss = loss

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_batch(self, data_loader, device):
        pass

    @abstractmethod
    def train(self, data_loader, epochs, lr, device, model_path=None, weight_decay=1e-6, with_gf=False):
        pass
