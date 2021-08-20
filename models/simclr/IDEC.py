from models.abstract_model.models import AbstractDecModel
from models.simclr.simclr import *
import torch


class IDEC(AbstractDecModel):
    def __init__(self, model=SimCLR(resnet_model='resnet50'), train_loader=None, device='cpu', n_clusters=None,
                 dec_type='IDEC', cluster_centres=torch.rand(size=(10, 2048))):
        """
        IDEC with SimCLR base.

            Parameters:
                model (SimCLR): SimCLR model to be used as an IDEC's base
                train_loader (Dataloader): data loader with data to be used for initial K-Means clustering
                device (str): device's name where data should be processed
                n_clusters: number of clusters K-Means should cluster the data into
                dec_type (str): 'IDEC' or 'DEC'
                cluster_centres: tensor containing cluster centres; required for IDEC
            Returns:
                IDEC SimCLR model
        """
        super().__init__(train_loader=train_loader, model=model, device=device, n_clusters=n_clusters,
                         dec_type=dec_type, cluster_centres=cluster_centres)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=1e-6, gf=False, write_stats=True,
            degree_of_space_distortion=0.1, dec_factor=0.1, with_aug=False):
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
                degree_of_space_distortion (float): weight controlling the impact of IDEC's loss
                dec_factor (float): factor at which the provided learning rate should be reduced; to be used if provided
                learning rate equals to the original's SimCLR rate
                with_aug (Boolean): if True, IDEC tries to assign augmented images view to the clusters predicted for
                the original image view
            Returns:
                model (IDEC): trained model
        """
        lr = start_lr * dec_factor
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        i = 0

        for epoch in range(epochs):
            for step, ((x, x_i, x_j), _) in enumerate(data_loader):
                i += 1
                x = x.to(device)
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                optimizer.zero_grad()

                if with_aug:
                    feats_i, mapped_feats_i = self.model(x_i)
                    feats_j, mapped_feats_j = self.model(x_j)
                    feats, _ = self.model(x)

                    base_loss = self.loss(mapped_feats_i, mapped_feats_j)
                    cluster_loss = self.cluster_module.loss_dec_compression(feats, [feats_i, feats_j])
                else:
                    _, mapped_feats_i = self.model(x_i)
                    _, mapped_feats_j = self.model(x_j)
                    feats, _ = self.model(x)

                    base_loss = self.loss(mapped_feats_i, mapped_feats_j)
                    cluster_loss = self.cluster_module.loss_dec_compression(feats)

                loss = base_loss + degree_of_space_distortion * cluster_loss
                # loss = cluster_loss

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
                    torch.save(self.state_dict(), model_path)

        return self
