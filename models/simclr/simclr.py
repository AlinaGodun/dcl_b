import torch
import torchvision
import numpy as np
from torch import nn
from models.simclr.cifar_resnets import ResNet18, ResNet50
from models.simclr.loss import SimCLRLoss
from models.abstract_model.models import AbstractModel
from util.gradflow_check import plot_grad_flow


class SimCLR(AbstractModel):
    def __init__(self, output_dim=128, resnet_model='resnet50', tau=0.5):
        super().__init__(name='SimCLR', loss=SimCLRLoss(tau))

        self.base_encoder = self.get_base_encoder(resnet_model)

        input_dim = self.base_encoder.fc.in_features

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )
        self.base_encoder.fc = nn.Sequential()

    def get_base_encoder(self, resnet_model):
        resnet_models = {
            'cifar_resnet18': ResNet18,
            'cifar_resnet50': ResNet50,
            'resnet18': torchvision.models.resnet18,
            'resnet50': torchvision.models.resnet50,
        }
        if resnet_model not in resnet_models.keys():
            raise KeyError(f'Provided resnet model: {resnet_model} is not available. \
             Available resnet models: {resnet_models}')
        return resnet_models[resnet_model]()

    def forward(self, x):
        feats = self.base_encoder(x)
        mapped_feats = self.projection_head(feats)
        return feats, mapped_feats

    def forward_batch(self, data_loader, device):
        embeddings = []
        labels = []
        for batch, batch_labels in data_loader:
            batch_data = batch.to(device)
            feats, _ = self(batch_data)
            embeddings.append(feats.detach().cpu())
            labels = labels + batch_labels.tolist()
        return torch.cat(embeddings, dim=0).numpy(), np.array(labels)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=1e-6, gf=False, write_stats=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr, weight_decay=weight_decay)
        i = 0

        for epoch in range(epochs):
            for step, ((x_i, x_j), _) in enumerate(data_loader):
                i += 1
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                optimizer.zero_grad()

                _, mapped_feats_i = self(x_i)
                _, mapped_feats_j = self(x_j)

                loss = self.loss(mapped_feats_i, mapped_feats_j)

                loss.backward()
                if gf:
                    plot_grad_flow(self.named_parameters())
                optimizer.step()

                self.iteration_stats.append(f'{epoch},{i},{loss.item():.4f}')
            self.epoch_stats.append(f'{epoch},{i},{loss.item():.4f}')

            if epoch % 5 == 0:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{loss.item():.4f},",
                      f"LR: {optimizer.param_groups[0]['lr']}")
                if model_path is not None:
                    torch.save(self.state_dict(), model_path)

        if write_stats:
            ew, iw = self.init_statistics()
            self.write_statistics(ew, 'epoch')
            self.write_statistics(iw, 'iteration')
            ew.close()
            iw.close()

        return self
