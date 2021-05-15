from models.abstract_model.models import AbstractDecModel
from models.rotnet.rotnet import RotNet
import torch

from util.gradflow_check import plot_grad_flow


class IDEC(AbstractDecModel):
    def __init__(self, train_loader, model=RotNet(num_classes=4, num_blocks=4), device='cpu', dec_type='DEC'):
        super().__init__(train_loader=train_loader, model=model, device=device, dec_type=dec_type)
        self.model = model

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=5e-4, gf=False, write_stats=True,
            degree_of_space_distortion=0.1, idec_factor=0.1):
        lr = start_lr * idec_factor
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        i = 0

        for epoch in range(epochs):
            for step, (x, labels) in enumerate(data_loader):
                i += 1
                x = x.to(device)
                # labels = labels.to(device)

                optimizer.zero_grad()

                # classifier_feats = self.model(x)
                feats = self.model(x, ['conv2']).flatten(start_dim=1)

                # base_loss = self.loss(classifier_feats, labels)
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
