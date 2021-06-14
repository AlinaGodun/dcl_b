from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional
import torch
import numpy as np
from sklearn.decomposition import PCA

from models.abstract_model.models import AbstractModel
from util.gradflow_check import plot_grad_flow


class RotNet(AbstractModel):
    def __init__(self, num_classes, in_channels=3, num_blocks=3, num_clusters=10):
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

        main_blocks += [RotNetGlobalAveragePooling()]
        # main_blocks += [nn.Linear(n_channels[1], num_clusters)]
        # main_blocks += [nn.Linear(num_clusters, num_classes)]
        main_blocks += [nn.Linear(n_channels[1], num_classes)]

        # main_blocks.append(nn.Sequential(OrderedDict([
        #     ('GlobalAveragePooling', RotNetGlobalAveragePooling()),
        #     ('Features', nn.Linear(num_channels[1], num_clusters)),
        #     ('Classifier', nn.Linear(n_channels[1], num_classes))
        # ])))

        self.feat_blocks = nn.ModuleList(main_blocks)
        # self.feat_block_names = [f'conv{s+1}' for s in range(num_blocks)] + ['pooling'] + ['features'] + ['classifier']
        self.feat_block_names = [f'conv{s + 1}' for s in range(num_blocks)] + ['pooling'] + ['classifier']

    def forward(self, x, layer='classifier'):
        if layer not in self.feat_block_names:
            raise KeyError(f'Provided layer: {layer} is not available. Available layers: {self.feat_block_names}')

        layer_index = self.feat_block_names.index(layer)

        feats = x
        for i in range(layer_index + 1):
            feats = self.feat_blocks[i](feats)

        return feats

    def forward_batch(self, data_loader, device, flatten=True, layer='conv2'):
        embeddings = []
        labels = []
        # pca = PCA(n_components=512)
        for batch, batch_labels in data_loader:
            batch_data = batch.to(device)
            feats = self(batch_data, layer)

            if flatten:
                feats = feats.flatten(start_dim=1)
                # feats = pca.fit_transform(feats.detach().cpu().numpy())
                # feats = torch.from_numpy(feats).to(device)

            embeddings.append(feats.detach().cpu())
            labels = labels + batch_labels.tolist()
        return torch.cat(embeddings, dim=0).numpy(), np.array(labels)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=5e-4, gf=False, write_stats=True):
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

        if write_stats:
            ew, iw = self.init_statistics()
            self.write_statistics(ew, self.epoch_stats)
            self.write_statistics(iw, self.iteration_stats)
            ew.close()
            iw.close()

        return self


class RotNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RotNetBasicBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RotNetGlobalAveragePooling(nn.Module):
    def __init__(self):
        super(RotNetGlobalAveragePooling, self).__init__()

    def forward(self, x):
        out_channels = x.size(1)
        kernel_size = (x.size(2), x.size(3))
        pooling = nn.functional.avg_pool2d(x, kernel_size)
        return pooling.view(-1, out_channels)
