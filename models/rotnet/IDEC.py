from models.abstract_model.models import AbstractDecModel
from models.rotnet.rotnet import RotNet
from util.pytorchtools import EarlyStopping
import numpy as np
import torch

from util.gradflow_check import plot_grad_flow


class IDEC(AbstractDecModel):
    def __init__(self, model=RotNet(num_classes=4, num_blocks=3), train_loader=None, device='cpu', dec_type='DEC',
                 cluster_centres=torch.rand(size=(4, 12288))):
        super().__init__(model=model, train_loader=train_loader, device=device, dec_type=dec_type,
                         cluster_centres=cluster_centres)

    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay=5e-4, gf=False, write_stats=True,
            degree_of_space_distortion=0.1, dec_factor=0.1, with_aug=False, eval_data_loader=None):
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
            for step, (x, labels) in enumerate(data_loader):
                i += 1
                x = x.to(device)

                optimizer.zero_grad()

                feats = self.model(x, 'conv2').flatten(start_dim=1)

                loss = self.cluster_module.loss_dec_compression(feats)

                optimizer.zero_grad()
                loss.backward()

                if gf:
                    plot_grad_flow(self.named_parameters())

                optimizer.step()

                train_losses.append(loss.item())
                self.iteration_stats.append(f'{epoch},{i},{loss.item():.4f}')

            if eval_data_loader is not None:
                with torch.no_grad():
                    for x, labels in eval_data_loader:
                        x = x.to(device)
                        feats = self.model(x, 'conv2').flatten(start_dim=1)
                        loss = self.cluster_module.loss_dec_compression(feats)
                        valid_losses.append(loss.item())

            self.epoch_stats.append(f'{epoch},{i},{loss.item():.4f}')

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_losses = []
            valid_losses = []

            if epoch % 5 == 0:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{train_loss:.4f}",
                      f"Validation loss:{valid_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
                # if model_path is not None:
                #     torch.save(self.state_dict(), model_path)

            early_stopping(valid_loss, self)

            if early_stopping.early_stop:
                break

        return self



