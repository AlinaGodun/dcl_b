from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional
import torch
import numpy as np

from models.abstract_model.models import AbstractModel
from util.gradflow_check import plot_grad_flow


class RotNet(AbstractModel):
    def __init__(self, num_classes=4, in_channels=3, num_blocks=5, num_clusters=10, with_features=False):
        """
        Implementation of Unsupervised Representation Learning by PredictingImage Rotations (RotNet):

            Parameters:
                num_classes (int): number of classes the images should be classified into; default values corresponds to the
                number of rotations; corresponds to the dimension of the output of the last classifier layer
                in_channels (int): number of input channels
                num_blocks (int): number of ConvNet blocks; recommended value is from 3 to 5
                num_clusters (int): number of clusters the images should be clustered into; corresponds to the dimension
                of the output of the features layer which is added if with_features is enabled; was used to try
                to cluster the data before classifying the rotation to use this output for IDEC training,
                but unsuccessfull
                with_features (Boolean): if True, include features layer; see num_clusters param for more info

            Returns:
                RotNet model
        """
        super().__init__(name='RotNet', loss=nn.CrossEntropyLoss())

        n_channels = {1: 192, 2: 160, 3: 96}

        main_blocks = [nn.Sequential(OrderedDict([
            ('B1_ConvB1', RotNetBasicBlock(in_channels=in_channels, out_channels=n_channels[1], kernel_size=5)),
            ('B1_ConvB2', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[2], kernel_size=1)),
            ('B1_ConvB3', RotNetBasicBlock(in_channels=n_channels[2], out_channels=n_channels[3], kernel_size=1)),
            ('B1_MaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ])), nn.Sequential(OrderedDict([
            ('B2_ConvB1', RotNetBasicBlock(in_channels=n_channels[3], out_channels=n_channels[1], kernel_size=5)),
            ('B2_ConvB2', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1)),
            ('B2_ConvB3', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1)),
            ('B2_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        ])), nn.Sequential(OrderedDict([
            ('B3_ConvB1', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=3)),
            ('B3_ConvB2', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1)),
            ('B3_ConvB3', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1))
        ]))]

        additional_blocks = [nn.Sequential(OrderedDict([
            (f'B{b+1}_ConvB1', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=3)),
            (f'B{b+1}_ConvB2', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1)),
            (f'B{b+1}_ConvB3', RotNetBasicBlock(in_channels=n_channels[1], out_channels=n_channels[1], kernel_size=1)),
        ])) for b in range(3, num_blocks)]

        main_blocks += additional_blocks

        if with_features:
            main_blocks += [RotNetGlobalAveragePooling()]
            main_blocks += [nn.Linear(n_channels[1], num_clusters)]
            main_blocks += [nn.Linear(num_clusters, num_classes)]

            self.feat_block_names = [f'conv{s+1}' for s in range(num_blocks)]
            self.feat_block_names += ['pooling'] + ['features'] + ['classifier']
        else:
            main_blocks.append(nn.Sequential(OrderedDict([
                ('GlobalAveragePooling', RotNetGlobalAveragePooling()),
                ('Classifier', nn.Linear(n_channels[1], num_classes))
            ])))

            self.feat_block_names = [f'conv{s + 1}' for s in range(num_blocks)] + ['classifier']

        self.feat_blocks = nn.ModuleList(main_blocks)

    def forward(self, x, layer='classifier'):
        """
        Forward input x through the model

            Parameters:
                x (tensor): input to be feed through the model
                layer (str): output of which layer should be returned; available layer names are available at
                rotnet_model.feat_block_names

            Returns:
                output of the provided layer

            Raises:
                KeyError: If layer value is not in the model.feat_block_names
        """
        if layer not in self.feat_block_names:
            raise KeyError(f'Provided layer: {layer} is not available. Available layers: {self.feat_block_names}')

        layer_index = self.feat_block_names.index(layer)

        feats = x
        for i in range(layer_index + 1):
            feats = self.feat_blocks[i](feats)

        return feats

    def forward_batch(self, data_loader, device, flatten=True, layer='conv2'):
        """
        Forward data provided by the data_loader batchwise

            Parameters:
                data_loader (Dataloader): dataloder providing data to be forwarded
                device (str): name of the device on which the data should be processed
                flatten (Boolean): if True, flatten the output of the layer; depends on which layer is used for
                forwaring; required if provided layer is ConvNet and output is used for k-means
                layer (str): output of which layer should be returned; available layer names are available at
                rotnet_model.feat_block_names

            Returns:
                output of the provided layer
        """
        embeddings = []
        labels = []

        for batch, batch_labels in data_loader:
            batch_data = batch.to(device)
            feats = self(batch_data, layer)

            if flatten:
                feats = feats.flatten(start_dim=1)

            embeddings.append(feats.detach().cpu())
            labels = labels + batch_labels.tolist()
        return torch.cat(embeddings, dim=0).numpy(), np.array(labels)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=5e-4, gf=False, write_stats=True):
        """
        Train model. Automatically saves model at the provided model_path.

            Parameters:
                data_loader (Dataloader): dataloder providing data to be forwarded
                epochs (int): number of epochs the model should be trained for
                start_lr (float): training's learning rate
                device (str): device's name for training
                model_path (str): path at which the model should be saved
                weight_decay (float): training's weight decay
                gf (Boolean): if True, plot gradient flow
                write_stats (Boolean): if True, write training statistics
            Returns:
                model (RotNet): trained model
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=start_lr,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 45], gamma=0.2)
        i = 0

        for epoch in range(epochs):
            for step, (x, labels) in enumerate(data_loader):
                i += 1
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                feats = self(x)
                loss = self.loss(feats, labels)

                loss.backward()
                if gf:
                    plot_grad_flow(self.named_parameters())
                optimizer.step()

                self.iteration_stats.append(f'{epoch},{i},{loss.item():.4f}')
            self.epoch_stats.append(f'{epoch},{i},{loss.item():.4f}')
            scheduler.step()

            if epoch % 5 == 0:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{loss.item():.4f},",
                      f"LR: {optimizer.param_groups[0]['lr']}")
                if model_path is not None:
                    torch.save(self.state_dict(), model_path)

        return self


class RotNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        A small ConvNet usead as a building block required for Net-In-Net architecture.

            Parameters:
                in_channels (int): number of the input channels
                out_channels (int): number of the output channels
                kernel_size (int): size of the kernel filter of the CNN
            Returns:
                ConvNet
        """
        super(RotNetBasicBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward input x through the model

            Parameters:
                x (tensor): input to be feed through the model
        """
        return self.block(x)


class RotNetGlobalAveragePooling(nn.Module):
    def __init__(self):
        super(RotNetGlobalAveragePooling, self).__init__()

    def forward(self, x):
        out_channels = x.size(1)
        kernel_size = (x.size(2), x.size(3))
        pooling = nn.functional.avg_pool2d(x, kernel_size)
        return pooling.view(-1, out_channels)
