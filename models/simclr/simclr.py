import torch
import torchvision
import cifar_resnets
from torch import nn
from simple_contrastive_loss import SimpleContrastiveLoss

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

    def fit(self, train_loader, epochs, start_lr, device, writer, tau=0.5, model_path=None, weight_decay=1e-5):
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr) # weight_decay ?
        loss = SimpleContrastiveLoss(tau)
        i = 0
        for epoch in epochs:
            for step, ((x_i, x_j), _) in enumerate(train_loader):
                i += 1
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                optimizer.zero_grad()

                _, mapped_feats_i = self(x_i)
                _, mapped_feats_j = self(x_j)

                loss = loss(mapped_feats_i, mapped_feats_j)

                loss.backward()
                optimizer.step()

                if epoch % 5 == 0 and model_path is not None:
                    print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{loss.item():.4f},",
                          f"LR: {optimizer.param_groups[0]['lr']}")
                    torch.save(self.state_dict(), model_path)
        return self

