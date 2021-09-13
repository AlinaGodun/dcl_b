import torch
import torchvision
import numpy as np
from torch import nn

from models.simclr.custom_cifar import SimCLRCIFAR
from models.simclr.custom_fmnist import SimCLRFMNIST
from models.simclr.custom_stl10 import SimCLRSTL10
from models.simclr.loss import SimCLRLoss
from models.abstract_model.models import AbstractModel
from util.gradflow_check import plot_grad_flow


class SimCLRFashionMNIST(object):
    pass


class SimCLR(AbstractModel):
    def __init__(self, output_dim=128, resnet_model='resnet50', tau=0.5):
        """
        Implementation of A Simple Framework for Contrastive Learning of Visual Representations (SimCLR):

            Parameters:
                output_dim (int): dimension of the output which should be provided by the model.
                resnet_model (str): which resnet model should be used as a base; available options are: cifar_resnet18
                (CIFAR-adapted ResNet-18), cifar_resnet50 (CIFAR-adapted ResNet-50), resnet18 (normal ResNet-18,
                standard pytorch implementation),resnet18 (normal ResNet-50, standard pytorch implementation).
                tau (float): temperature parameter; recommended values are in range from 0.1 to 1.0, CIFAR recommended
                value is 0.5.

            Returns:
                SimCLR model

            Raises:
                KeyError: If resnet_model value is not in the list of available options
        """
        super().__init__(name='SimCLR', loss=SimCLRLoss(tau))

        self.datasets = {
            'cifar': SimCLRCIFAR,
            'stl10': SimCLRSTL10,
            'fmnist': SimCLRFMNIST
        }

        self.resnet_models = {
            'resnet18': torchvision.models.resnet18,
            'resnet50': torchvision.models.resnet50,
        }

        self.base_encoder = self.get_base_encoder(resnet_model)

        input_dim = self.base_encoder.fc.in_features

        self.base_encoder.fc = nn.Sequential()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def get_base_encoder(self, resnet_model):
        """
        Get resnet which should be used as a base encoder for the SimCLR

            Parameters:
                resnet_model (str): which resnet model should be used as a base; available options are: resnet18
                (normal ResNet-18, standard pytorch implementation), resnet18 (normal ResNet-50, standard pytorch
                implementation).

            Returns:
                Base encoder

            Raises:
                KeyError: If resnet_model value is not in the list of available options
        """
        if resnet_model not in self.resnet_models.keys():
            raise KeyError(f'Provided resnet model: {resnet_model} is not available. \
             Available resnet models: {self.resnet_models.keys()}')
        return self.resnet_models[resnet_model]()

    def forward(self, x):
        """
        Forward input x through the model

            Parameters:
                x (tensor): input to be feed through the model

            Returns:
                (feats, mapped_feats) - where feats is a vector of relevant features which could be used for downstream
                tasks and mapped_feats is the feature vector mapped to the space where SimCLR's contrasive loss is
                applied
        """
        feats = self.base_encoder(x)
        mapped_feats = self.projection_head(feats)
        return feats, mapped_feats

    def forward_batch(self, data_loader, device, flatten=None, with_aug=False):
        """
        Forward data provided by the data_loader batchwise

            Parameters:
                data_loader (DataLoader): dataloder providing data to be forwarded
                device (str): name of the device on which the data should be processed
                flatten (Boolean): this argument is not used in SimCLR; it is present because SimCLR is a subclass of
                AbstractModel, and other models need this argument

            Returns:
                (forwarded_data, labels, augmented_labels) - where forwarded data is a data forwarded through the model,
                labels contain ground truth labels and augmented_labels are set to 1 if data is augmented and 0 if not
        """
        embeddings = []
        labels = []
        aug_labels = []

        for batch, batch_labels in data_loader:
            batch_data = batch.to(device)
            feats, _ = self(batch_data)
            embeddings.append(feats.detach().cpu())

            if with_aug:
                labels = labels + batch_labels[0].tolist()
                aug_labels = aug_labels + batch_labels[1].tolist()
            else:
                labels = labels + batch_labels.tolist()

        if with_aug:
            return torch.cat(embeddings, dim=0).numpy(), np.array(labels), np.array(aug_labels)
        else:
            return torch.cat(embeddings, dim=0).numpy(), np.array(labels)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=1e-6, gf=False, write_stats=True):
        """
        Train model. Automatically saves model at the provided model_path.

            Parameters:
                data_loader (DataLoader): dataloder providing data to be forwarded
                epochs (int): number of epochs the model should be trained for
                start_lr (float): training's learning rate
                device (str): device's name for training
                model_path (str): path at which the model should be saved
                weight_decay (float): training's weight decay
                gf (Boolean): if True, plot gradient flow
                write_stats (Boolean): if True, write training statistics
            Returns:
                model (SimCLR): trained model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr, weight_decay=weight_decay)

        i = 0

        for epoch in range(epochs):
            self.train()
            for step, ((x_i, x_j), _) in enumerate(data_loader):
                i += 1
                # load data to device
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                # get feats mapped to the space for simclr loss for augmented images
                _, mapped_feats_i = self(x_i)
                _, mapped_feats_j = self(x_j)

                loss = self.loss(mapped_feats_i, mapped_feats_j)

                optimizer.zero_grad()
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
                    self.eval()
                    torch.save(self.state_dict(), model_path)

        return self
