# Based on  https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import numpy as np

from models.abstract_model.models import AbstractModel
from models.autoencoder.custom_cifar import AECIFAR
from models.autoencoder.custom_fmnist import AEFMNIST
from models.autoencoder.custom_stl10 import AESTL10
from util.pytorchtools import EarlyStopping


class ConvAE(AbstractModel):
    def __init__(self, n_channels=3, n_classes=3, embd_sz=64):
        """
        Implementation of a convolution autoencoder

            Parameters:
                n_channels (int): number of input channels
                n_classes (int): number of classes; for images, must correspond to the number of channels
                embd_sz (int): size of the embedding (encoder's output and decoder's input)
            Returns:
                Autoencoder model
        """
        super().__init__(name='AE', loss=nn.MSELoss())
        self.datasets = {
            'cifar': AECIFAR,
            'stl10': AESTL10,
            'fmnist': AEFMNIST
        }

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBn(n_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        factor = 1
        self.down3 = Down(512, 1024 // factor)
        enc_dim = 1024 * 4 * 4
        self.flatten = Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(enc_dim, embd_sz*2),
            nn.BatchNorm1d(embd_sz*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embd_sz*2, embd_sz),
        )
        self.decoder = nn.Sequential(nn.Linear(embd_sz, enc_dim),
                                     nn.BatchNorm1d(enc_dim),
                                     nn.LeakyReLU(inplace=True),
                                     )
        self.up1 = Up(1024, 512 // factor, scale_factor=2)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)

        self.outc = OutConv(128, n_classes)

    def encode(self, x):
        """
        Embed input x with autoencoder's encoder

            Parameters:
                x (tensor): x to be encoded

            Returns:
                encoded x
        """
        e = self.inc(x)
        e = self.down1(e)
        e = self.down2(e)
        e = self.down3(e)
        e = self.flatten(e)
        e = self.encoder(e)
        return e

    def decode(self, e):
        """
        Reconstruct input x from the encoding e with autoencoder's decoder

            Parameters:
                e (tensor): encoding to be reconstructed

            Returns:
                reconstruction of e
        """
        d = self.decoder(e)
        d = d.view(-1, 1024, 4,4)
        d = self.up1(d)
        d = self.up2(d)
        d = self.up3(d)
        return self.outc(d)

    def forward(self, x):
        """
        Forward input x through the model (encoder and decoder)

            Parameters:
                x (tensor): input to be feed through the model

            Returns:
                reconstruction of x
        """
        e = self.encode(x)
        d = self.decode(e)
        return d

    def forward_batch(self, data_loader, device, flatten=None):
        self.eval()
        """
        Forward data provided by the data_loader batchwise

            Parameters:
                data_loader (DataLoader): dataloder providing data to be forwarded
                device (str): name of the device on which the data should be processed
                flatten (Boolean): this argument is not used in Autoencoder; it is present because Autoencoder is a
                subclass of AbstractModel, and other models need this argument

            Returns:
                (forwarded_data, labels, augmented_labels) - where forwarded data is a data forwarded through the model,
                labels contain ground truth labels and augmented_labels are set to 1 if data is augmented and 0 if not
        """
        embeddings = []
        labels = []

        for batch, batch_labels in data_loader:
            batch_data = batch.to(device)
            encoding = self.encode(batch_data)
            embeddings.append(encoding.detach().cpu())
            labels = labels + batch_labels.tolist()

        return torch.cat(embeddings, dim=0).numpy(), np.array(labels)

    def fit(self, data_loader, epochs, start_lr, device, model_path=None, weight_decay=1e-5, eval_data_loader=None):
        """
        Train model. Automatically saves model at the provided model_path.

            Parameters:
                data_loader (DataLoader): dataloder providing data to be forwarded
                epochs (int): number of epochs the model should be trained for
                start_lr (float): training's learning rate
                device (str): device's name for training
                model_path (str): path at which the model should be saved
                weight_decay (float): training's weight decay
                eval_data_loader (DataLoader): dataloader providing data for evaluation; if not None, early stopping
                is used: if validation loss from the evaluation dataset does not decrease for 10 epochs, training stops
            Returns:
                model (SimCLR): trained model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr, weight_decay=weight_decay)
        i = 0

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

        for epoch_i in range(epochs):
            self.train()
            for batch_data, _ in data_loader:
                # load batch on device
                batch = batch_data.to(device)

                # reset gradients from last iteration
                optimizer.zero_grad()
                reconstruction = self(batch)
                loss = self.loss(reconstruction, batch)
                # calculate gradients and reset the computation graph
                loss.backward()
                # update the internal params (weights, etc.)
                optimizer.step()
                i += 1
                train_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_losses = []
            valid_losses = []

            if eval_data_loader is not None:
                with torch.no_grad():
                    self.eval()
                    for x, labels in eval_data_loader:
                        x = x.to(device)
                        reconstruction = self(x)
                        loss = self.loss(reconstruction, x)
                        valid_losses.append(loss.item())

            if epoch_i % 5 == 0 and model_path is not None:
                print(f"{self.name}: Epoch {epoch_i + 1}/{epochs} - Iteration {i} - Train loss:{train_loss:.4f}",
                      f"Validation loss:{valid_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
                if model_path is not None:
                    self.eval()
                    torch.save(self.state_dict(), model_path)

            if eval_data_loader is not None:
                early_stopping(valid_loss, self)

                if early_stopping.early_stop:
                    break

        return self


class ConvBn(nn.Module):
    """(convolution => [BN] => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn(x)


class ConvLeaky(nn.Module):
    """(convolution => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling double conv and stride"""

    def __init__(self, in_channels, out_channels, kernel_size=3, bn=True):
        super().__init__()
        if bn:
            self.conv = ConvBn(in_channels, out_channels, stride=2, kernel_size=kernel_size)
        else:
            self.conv = ConvLeaky(in_channels, out_channels, stride=2, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, bn=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        if bn:
            self.conv = ConvBn(in_channels , out_channels, kernel_size=kernel_size)
        else:
            self.conv = ConvLeaky(in_channels , out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    """From fastai library:
    Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"""
    def __init__(self, full: bool = False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)