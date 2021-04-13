from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional
import torch
from util.gradflow_check import plot_grad_flow


class RotNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_blocks=3):
        super(RotNet, self).__init__()

        self.name = 'RotNet'
        num_channels = {1: 192, 2: 160, 3: 96}

        main_blocks = [nn.Sequential(OrderedDict([
            ('B1_ConvB1', RotNetBasicBlock(in_channels=in_channels, out_channels=num_channels[1], kernel_size=5)),
            ('B1_ConvB2', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[2], kernel_size=1)),
            ('B1_ConvB3', RotNetBasicBlock(in_channels=num_channels[2], out_channels=num_channels[3], kernel_size=1)),
            ('B1_MaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ])), nn.Sequential(OrderedDict([
            ('B2_ConvB1', RotNetBasicBlock(in_channels=num_channels[3], out_channels=num_channels[1], kernel_size=5)),
            ('B2_ConvB2', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1)),
            ('B2_ConvB3', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1)),
            ('B2_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        ])), nn.Sequential(OrderedDict([
            ('B3_ConvB1', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=3)),
            ('B3_ConvB2', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1)),
            ('B3_ConvB3', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1))
        ]))]

        additional_blocks = [nn.Sequential(OrderedDict([
            (f'B{b+1}_ConvB1', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=3)),
            (f'B{b+1}_ConvB2', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1)),
            (f'B{b+1}_ConvB3', RotNetBasicBlock(in_channels=num_channels[1], out_channels=num_channels[1], kernel_size=1)),
        ])) for b in range(3, num_blocks)]

        main_blocks += additional_blocks

        main_blocks.append(nn.Sequential(OrderedDict([
            ('GlobalAveragePooling', RotNetGlobalAveragePooling()),
            ('Classifier', nn.Linear(num_channels[1], num_classes))
        ])))

        self.feat_blocks = nn.ModuleList(main_blocks)
        self.feat_block_names = [f'conv{s+1}' for s in range(num_blocks)] + ['classifier']

    def parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = [self.feat_block_names[-1]] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('No output feature keys given.')
        for key in out_feat_keys:
            if key not in self.feat_block_names:
                raise ValueError(
                    f'Provided feature name: {key} does not exist. Existing feature names: {self.feat_block_names}.')

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.feat_block_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = self.parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for i in range(max_out_feat + 1):
            key = self.feat_block_names[i]
            feat = self.feat_blocks[i](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        return out_feats

    def fit(self, trainloader, epochs, start_lr, device, model_path=None, weight_decay=5e-4, with_gf=False):
        optimizer = torch.optim.SGD(self.parameters(), lr=start_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
        rotnet_loss = nn.CrossEntropyLoss()
        i = 0

        epoch_writer = open(f"epoch_stat_{self.name}.csv", "w")
        iteration_writer = open(f"iteration_stat_{self.name}.csv", "w")

        epoch_losses = ['epoch,iteration,loss']
        iteration_losses = ['epoch,iteration,loss']

        for epoch in range(epochs):
            for step, (x, labels) in enumerate(trainloader):
                i += 1
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # print(x.shape)
                feats = self(x)
                # print(feats.shape)
                loss = rotnet_loss(feats, labels)

                loss.backward()
                if with_gf:
                    plot_grad_flow(self.named_parameters())

                optimizer.step()

                iteration_losses.append(f'{epoch},{i},{loss.item():.4f}')
            scheduler.step()

            epoch_losses.append(f'{epoch},{i},{loss.item():.4f}')

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

    def forward(self, feat):
        return torch.nn.functional.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, feat.size(1))
