import torch
import torch.nn as nn


class SimCLRLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(SimCLRLoss, self).__init__()
        self.tau = tau

    def get_sim_matrix(self, x1, x2):
        nom = x1.mm(x2.t())
        batch_norm = torch.linalg.norm(x1, dim=1)
        au_batch_norm = torch.linalg.norm(x2, dim=1)
        denom = batch_norm.unsqueeze(1) * au_batch_norm * self.tau
        return nom/denom.clamp(min=1e-16)

    def forward(self, x1, x2):
        c1 = torch.cat((x1, x2), 0)
        c2 = torch.cat((x2, x1), 0) ## detach? look closer: tensor from same tensors
        sim_pairwise = torch.exp(self.get_sim_matrix(c1, c2)) ## replace with cosinesimilarity?

        nom = sim_pairwise.diag()
        denom = torch.sum(sim_pairwise, dim=0) - nom
        positive_losses = -torch.log(nom/denom.clamp(1e-16))
        return positive_losses.mean()



