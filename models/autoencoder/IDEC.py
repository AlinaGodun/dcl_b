from models.abstract_model.models import AbstractDecModel
from models.autoencoder.conv_ae import ConvAE
import torch
import numpy as np

from util.pytorchtools import EarlyStopping


class IDEC(AbstractDecModel):
    def __init__(self, model=ConvAE(n_channels=3, n_classes=3), train_loader=None, device='cpu', n_clusters=None,
                 dec_type='IDEC', cluster_centres=torch.rand(size=(10, 128))):
        """
        DEC with ConvAE base.

            Parameters:
                model (ConvAE): ConvAE model to be used as an DEC's base
                train_loader (DataLoader): data loader with data to be used for initial K-Means clustering
                device (str): device's name where data should be processed
                n_clusters: number of clusters K-Means should cluster the data into
                dec_type (str): 'IDEC' or 'DEC'
                cluster_centres: tensor containing cluster centres; required for DEC
            Returns:
                IDEC ConvAE model
        """
        super().__init__(train_loader=train_loader, model=model, device=device, n_clusters=n_clusters,
                         dec_type=dec_type, cluster_centres=cluster_centres)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=1e-6, gf=False, write_stats=True,
            degree_of_space_distortion=0.1, dec_factor=0.1, with_aug=False, eval_data_loader=None):
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.cluster_module.parameters()),
                                     lr=start_lr)

        early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        cluster_path = model_path.replace('.pth', '_cm.pth')

        i = 0

        for epoch in range(epochs):
            for batch in data_loader:
                batch_data = batch[0].to(device)
                embedded = self.model.encode(batch_data)
                reconstruction = self.model.decode(embedded)

                ae_loss = self.loss(batch_data, reconstruction)
                cluster_loss = self.cluster_module.loss_dec_compression(embedded)

                loss = ae_loss + degree_of_space_distortion * cluster_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            if eval_data_loader is not None:
                with torch.no_grad():
                    for x, labels in eval_data_loader:
                        x = x.to(device)
                        embedded = self.model.encode(x)
                        reconstruction = self.model.decode(embedded)

                        ae_loss = self.loss(x, reconstruction)
                        cluster_loss = self.cluster_module.loss_dec_compression(embedded)
                        loss = ae_loss + degree_of_space_distortion * cluster_loss

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
                    torch.save(self.cluster_module, cluster_path)

            early_stopping(valid_loss, self)

            if early_stopping.early_stop:
                break

        return self
