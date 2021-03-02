import torchvision
from torch import nn
import cifar_resnets


class SimCLR(nn.Module):
    def __init__(self, output_dim, resnet_model='cifar_resnet'):
        super().__init__()

        self.resnet_models = {
            'cifar_resnet18': cifar_resnets.ResNet18(),
            'cifar_resnet50': cifar_resnets.ResNet50(),
            'resnet18': torchvision.models.resnet18(),
            'resnet50': torchvision.models.resnet50(),
        }

        self.base_encoder = self.get_resnet(resnet_model)

        input_dim = self.base_encoder.fc.in_features
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )

        self.base_encoder.fc = nn.Sequential()

    def get_resnet(self, resnet_model):
        if resnet_model not in self.resnet_models.keys():
            raise KeyError(f"{resnet_model} is not in the list of available resnet models. \
                Available resnet models: {self.resnet.keys()}")
        return self.resnet_models[resnet_model]

    def forward(self, x):
        feats = self.base_encoder(x)
        mapped_feats = self.projection_head(feats)
        return feats, mapped_feats
