from abc import abstractmethod, ABC
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from models.dec.DEC import DEC


class AbstractModel(nn.Module):
    def __init__(self, name, loss, epoch_stats=None, it_stats=None):
        super(AbstractModel, self).__init__()
        self.name = name
        self.loss = loss

        self.epoch_stats = ['epoch,iteration,loss'] if epoch_stats is None else epoch_stats
        self.iteration_stats = ['epoch,iteration,loss'] if it_stats is None else it_stats

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_batch(self, data_loader, device):
        pass

    @abstractmethod
    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay, gf=False, write_stats=True):
        pass

    def init_statistics(self):
        statistics_path = f'statistics/{self.name}/'
        ew = open(f'{statistics_path}epoch_stat.csv', 'w')
        iw = open(f'{statistics_path}iteration_stat.csv', 'w')
        return ew, iw

    def write_statistics(self, writer, stat_list):
        stat = '\n'.join(map(str, stat_list))
        writer.write(stat)
        stat_list.clear()


class AbstractDecModel(AbstractModel, ABC):
    def __init__(self, train_loader, model, device='cpu', n_clusters=None, dec_type='IDEC'):
        super().__init__(self, f'{dec_type}_{model.name}', model.loss)

        if not issubclass(model, AbstractModel):
            raise TypeError(f'Model must inherit class AbstractModel')
        self.model = model.to(device)

        embedded_data, labels = self.forward_batch(train_loader, device=device)
        n_clusters = len(set(labels)) if n_clusters is None else n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embedded_data)

        self.start_nmi = normalized_mutual_info_score(labels, kmeans.labels_)
        self.cluster_module = DEC(kmeans.cluster_centers_).to(device)

    def forward(self, x):
        return self.model(x)

    def forward_batch(self, data_loader, device):
        return self.model.forward_batchwise(data_loader, device)
