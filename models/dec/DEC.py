import torch
from models.dec.util import dec_prediction
from models.dec.util import dec_compression_loss_fn


class DEC(torch.nn.Module):
    def __init__(self, init_np_centers, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_np_centers), requires_grad=True)

    def prediction(self, embedded)->torch.Tensor:
        """Soft prediction $q$"""
        return dec_prediction(self.centers, embedded, self.alpha)

    def prediction_hard(self, embedded)->torch.Tensor:
        """Hard prediction"""
        return self.prediction(embedded).argmax(1)

    def loss_dec_compression(self, embedded)->torch.Tensor:
        """Loss of DEC"""
        prediction = dec_prediction(self.centers, embedded, self.alpha)
        loss = dec_compression_loss_fn(prediction)
        return loss