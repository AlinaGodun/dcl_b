from models.abstract_model.models import AbstractDecModel
from models.simclr.simclr import *
import torch


class IDEC(AbstractDecModel):
    def __init__(self, model=SimCLR(resnet_model='resnet50'), train_loader=None, device='cpu', n_clusters=None,
                 dec_type='IDEC', cluster_centres=torch.rand(size=(10, 2048))):
        super().__init__(train_loader=train_loader, model=model, device=device, n_clusters=n_clusters,
                         dec_type=dec_type, cluster_centres=cluster_centres)
        self.model = model

        # set SimCLR ResNet to eval mode
        self.model.base_encoder.eval()

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=1e-6, gf=False, write_stats=True,
            degree_of_space_distortion=0.1, dec_factor=0.1):
        lr = start_lr * dec_factor
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=weight_decay)
        i = 0

        for epoch in range(epochs):
            for step, ((x, x_i, x_j), _) in enumerate(data_loader):
                i += 1
                x = x.to(device)
                # x_i = x_i.to(device)
                # x_j = x_j.to(device)

                optimizer.zero_grad()

                # _, mapped_feats_i = self.model(x_i)
                # _, mapped_feats_j = self.model(x_j)
                feats, _ = self.model(x)

                # base_loss = self.loss(mapped_feats_i, mapped_feats_j)
                cluster_loss = self.cluster_module.loss_dec_compression(feats)
                # loss = base_loss + degree_of_space_distortion * cluster_loss
                loss = cluster_loss

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

        if write_stats:
            ew, iw = self.init_statistics()
            self.write_statistics(ew, self.epoch_stats)
            self.write_statistics(iw, self.iteration_stats)
            ew.close()
            iw.close()

        return self
