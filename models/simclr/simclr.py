import torch
import torchvision
from torch import nn
from models.simclr.cifar_resnets import ResNet18, ResNet50
from models.simclr.loss import SimCLRLoss
from util.gradflow_check import plot_grad_flow

class SimCLR(nn.Module):
    def __init__(self, output_dim=128, resnet_model='cifar_resnet18'):
        super().__init__()
        self.name = 'SimCLR'
        self.resnet_models = {
            'cifar_resnet18': ResNet18(),
            'cifar_resnet50': ResNet50(),
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

    def fit(self, trainloader, epochs, start_lr, device, model_path=None, weight_decay=1e-6, tau=0.5, with_gf=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr, weight_decay=weight_decay)
        simclr_loss = SimCLRLoss(tau)
        i = 0

        epoch_writer = open("epoch_stat.csv", "w")
        iteration_writer = open("iteration_stat.csv", "w")

        epoch_losses = []
        iteration_losses = []
        for epoch in range(epochs):
            for step, ((x_i, x_j), _) in enumerate(trainloader):
                i += 1
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                optimizer.zero_grad()

                _, mapped_feats_i = self(x_i)
                _, mapped_feats_j = self(x_j)

                loss = simclr_loss(mapped_feats_i, mapped_feats_j)

                loss.backward()
                if with_gf:
                    plot_grad_flow(self.named_parameters())

                optimizer.step()

                iteration_losses.append(f'{epoch}, {i}, {loss.item():.4f}')

            epoch_losses.append(f'{epoch}, {i}, {loss.item():.4f}')

            if epoch % 5 == 0 and model_path is not None:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{loss.item():.4f},",
                      f"LR: {optimizer.param_groups[0]['lr']}")
                torch.save(self.state_dict(), model_path)

                stat = '\n'.join(map(str, epoch_losses))
                epoch_writer.write(stat)
                epoch_losses.clear()

                stat = '\n'.join(map(str, iteration_losses))
                iteration_writer.write(stat)
                iteration_losses.clear()

        epoch_writer.close()
        iteration_writer.close()

        return self
