from models.abstract_model.models import AbstractDecModel
from models.rotnet.rotnet import RotNet
from util.pytorchtools import EarlyStopping
import numpy as np
import torch

from util.gradflow_check import plot_grad_flow


class DEC(AbstractDecModel):
    def __init__(self, model=RotNet(num_classes=4, num_blocks=3), train_loader=None, device='cpu', dec_type='DEC',
                 cluster_centres=torch.rand(size=(4, 12288)), n_clusters=None):
        """
        DEC with RotNet base.

            Parameters:
                model (RotNet): RotNet model to be used as an DEC's base
                train_loader (DataLoader): data loader with data to be used for initial K-Means clustering
                device (str): device's name where data should be processed
                n_clusters: number of clusters K-Means should cluster the data into
                dec_type (str): 'IDEC' or 'DEC'
                cluster_centres: tensor containing cluster centres; required for DEC
            Returns:
                IDEC RotNet model
        """
        super().__init__(model=model, train_loader=train_loader, device=device, dec_type=dec_type,
                         cluster_centres=cluster_centres, n_clusters=n_clusters)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=5e-4, gf=False, write_stats=True,
            degree_of_space_distortion=None, dec_factor=0.1, with_aug=False, eval_data_loader=None):
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
                degree_of_space_distortion (float): weight controlling the impact of IDEC's loss; not used since DEC
                does not limit clustering loss
                dec_factor (float): factor at which the provided learning rate should be reduced; to be used if provided
                learning rate equals to the original's SimCLR rate
                with_aug (Boolean): if True, IDEC tries to assign augmented images view to the clusters predicted for
                the original image view
                eval_data_loader (DataLoader): dataloader providing data for evaluation; if not None, early stopping
                is used: if validation loss from the evaluation dataset does not decrease for 10 epochs, training stops
            Returns:
                model (IDEC): trained model
        """
        lr = start_lr * dec_factor
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=weight_decay)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

        i = 0
        for epoch in range(epochs):
            self.train()
            for step, (x, labels) in enumerate(data_loader):
                i += 1
                x = x.to(device)

                optimizer.zero_grad()

                feats = self.model(x, 'conv2').flatten(start_dim=1)
                loss = self.cluster_module.loss_dec_compression(feats)

                loss.backward()

                if gf:
                    plot_grad_flow(self.named_parameters())

                optimizer.step()

                train_losses.append(loss.item())
                self.iteration_stats.append(f'{epoch},{i},{loss.item():.4f}')
            self.epoch_stats.append(f'{epoch},{i},{loss.item():.4f}')

            if eval_data_loader is not None:
                with torch.no_grad():
                    for x, labels in eval_data_loader:
                        x = x.to(device)
                        feats = self.model(x, 'conv2').flatten(start_dim=1)
                        loss = self.cluster_module.loss_dec_compression(feats)
                        valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_losses = []
            valid_losses = []

            if epoch % 5 == 0:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{train_loss:.4f}",
                      f"Validation loss:{valid_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
                if model_path is not None:
                    self.eval()
                    torch.save(self.state_dict(), model_path)

            if eval_data_loader is not None:
                early_stopping(valid_loss, self)

                if early_stopping.early_stop:
                    break

        return self



