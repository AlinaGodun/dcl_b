from sklearn.cluster import KMeans
from models.dec import DEC
from util.util import encode_batchwise
from util.util import evaluate_batchwise
import torch.nn as nn
import torch
import os



class IDEC(nn.Module):
    def __init__(self, ae_model, testloader, device):
        self.device = device
        self.ae_model = ae_model
        self.ae_model.name = 'idec' + self.ae_model.name

        (embedded_data, labels) = encode_batchwise(testloader, ae_model, device)
        kmeans = KMeans(n_clusters=len(set(labels)))
        kmeans.fit(embedded_data)
        init_centers = kmeans.cluster_centers_

        self.cluster_module = DEC(init_centers).to(self.device)
        self.cluster_module.name = 'i' + self.cluster_module.name

    def forward(self, x):
        return self.ae_model(x)

    def train(self, epochs, trainloader, testloader, lr, base_path):
        degree_of_space_distortion = 0.1

        # Note: We now optimize the autoencoder and the DEC parameters jointly together
        optimizer = torch.optim.Adam(list(self.ae_model.parameters()) + list(self.cluster_module.parameters()), lr=lr)
        loss_fn = torch.nn.MSELoss()

        idec_model_path = os.path.join(base_path, self.ae_model.name)
        idec_path = os.path.join(base_path, self.cluster_module.name)

        for epoch in range(epochs):  # each iteration is equal to an epoch
            for batch in trainloader:
                batch_data = batch[0].to(self.device)
                embedded = self.ae_model.encode(batch_data)
                reconstruction = self.ae_model.decode(embedded)

                ae_loss = loss_fn(batch_data, reconstruction)
                cluster_loss = self.cluster_module.loss_dec_compression(embedded)

                # Reconstruction loss is now included
                # L = L_r + \gamma L_c
                loss = ae_loss + degree_of_space_distortion * cluster_loss
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    nmi = evaluate_batchwise(testloader, self.ae_model, self.cluster_module, self.device)
                    print(f"{epoch}/{epochs} cluster_loss:{cluster_loss.item():.4f} NMI:{nmi:.4f}"
                          f" ae_loss:{ae_loss.item():.4f} total_loss: {(ae_loss.item() + cluster_loss.item()):.4f}]")

        # save model
        torch.save(self.ae_model.state_dict(), idec_model_path)
        # save IDEC
        torch.save(self.cluster_module.state_dict(), idec_path)