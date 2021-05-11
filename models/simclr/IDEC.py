from models.dec.DEC import DEC
from models.simclr.simclr import *
import torch

class IDEC(torch.nn.Module):
    def __init__(self, model=SimCLR(resnet_model='resnet50'), loss=None, cluster_centers=torch.rand(size=(10, 2048)), device='cpu'):
        super().__init__()
        self.model = model
        self.name = f'IDEC_{model.name}'
        self.loss = loss
        self.cluster_module = DEC(cluster_centers).to(device)

        ## set resnet in simclr to eval mode
        self.model.base_encoder.eval()


    def fit(self, trainloader, epochs, start_lr, device, model_path=None, degree_of_space_distortion=0.1, idec_factor=0.1, weight_decay=1e-6):
        lr = start_lr * idec_factor
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        i = 0

        epoch_writer = open(f"epoch_stat_{self.name}.csv", "w")
        iteration_writer = open(f"iteration_stat_{self.name}.csv", "w")

        epoch_losses = ['epoch,iteration,loss']
        iteration_losses = ['epoch,iteration,loss']

        for epoch in range(epochs):
            for step, ((x, x_i, x_j), _) in enumerate(trainloader):
                i += 1
                x = x.to(device)
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                optimizer.zero_grad()

                _, mapped_feats_i = self.model(x_i)
                _, mapped_feats_j = self.model(x_j)
                feats, _ = self.model(x)

                base_loss = self.loss(mapped_feats_i, mapped_feats_j)
                cluster_loss = self.cluster_module.loss_dec_compression(feats)
                loss = base_loss + degree_of_space_distortion * cluster_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration_losses.append(f'{epoch},{i},{loss.item():.4f}')

            epoch_losses.append(f'{epoch},{i},{loss.item():.4f}')

            if epoch % 5 == 0 and model_path is not None:
                print(f"{self.name}: Epoch {epoch + 1}/{epochs} - Iteration {i} - Train loss:{loss.item():.4f},",
                      f"LR: {optimizer.param_groups[0]['lr']}")
                torch.save(self.state_dict(), model_path)

                stat = '\n'.join(map(str, epoch_losses))
                epoch_writer.write('\n' + stat)
                epoch_losses.clear()

                stat = '\n'.join(map(str, iteration_losses))
                iteration_writer.write('\n' + stat)
                iteration_losses.clear()

        epoch_writer.close()
        iteration_writer.close()

        return self
