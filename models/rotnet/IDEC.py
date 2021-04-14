from models.dec.DEC import DEC
from models.simclr.simclr import *
import torch

class IDEC(torch.nn.Module):
    def __init__(self, model=SimCLR(resnet_model='resnet18'), loss=None, cluster_centers=torch.rand(size=(10, 512)), device='cpu'):
        super().__init__()
        self.model = model
        self.name = f'IDEC_{model.name}'
        self.loss = loss
        self.cluster_module = DEC(cluster_centers).to(device)

    def fit(self, trainloader, epochs, start_lr, device, model_path=None, weight_decay=5e-4, with_gf=False, degree_of_space_distortion=0.1, idec_factor=0.1):
        lr = start_lr * idec_factor
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
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
                classifier_feats = self.model(x)
                feats = self.model(x, ['conv2']).flatten(start_dim=1)
                print(feats.shape)

                # print(feats.shape)
                base_loss = rotnet_loss(classifier_feats, labels)
                cluster_loss = self.cluster_module.loss_dec_compression(feats)
                loss = base_loss + degree_of_space_distortion * cluster_loss

                loss.backward()
                if with_gf:
                    plot_grad_flow(self.named_parameters())

                optimizer.step()

                iteration_losses.append(f'{epoch},{i},{loss.item():.4f}')

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