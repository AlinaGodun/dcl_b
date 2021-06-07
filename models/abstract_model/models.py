from abc import abstractmethod, ABC

import torch
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
    def forward_batch(self, data_loader, device, flatten=True):
        pass

    @abstractmethod
    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay, gf=False, write_stats=True):
        pass

    def init_statistics(self):
        return init_statistics(self.name)

    def write_statistics(self, writer, stat):
        write_statistics(writer, stat)


class AbstractDecModel(nn.Module):
    def __init__(self, model, train_loader=None, device='cpu', n_clusters=None, dec_type='IDEC',
                 cluster_centres=torch.rand(size=(4, 12288)), epoch_stats=None, it_stats=None):
        super(AbstractDecModel, self).__init__()
        self.name = f'{dec_type}_{model.name}'
        self.loss = model.loss

        self.epoch_stats = ['epoch,iteration,loss'] if epoch_stats is None else epoch_stats
        self.iteration_stats = ['epoch,iteration,loss'] if it_stats is None else it_stats

        if not issubclass(model.__class__, AbstractModel):
            raise TypeError(f'Model must inherit class AbstractModel. Model class is {type(model)}')
        self.model = model.to(device)
        self.model.eval()

        if train_loader is not None:
            embedded_data, labels = self.forward_batch(train_loader, device=device)
            n_clusters = len(set(labels)) if n_clusters is None else n_clusters
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(embedded_data)
            cluster_centres = kmeans.cluster_centers_
            self.start_nmi = normalized_mutual_info_score(labels, kmeans.labels_)

        self.cluster_module = DEC(cluster_centres).to(device)

    def forward(self, x):
        return self.model(x)

    def forward_batch(self, data_loader, device, flatten=True, layer=None):
        return self.model.forward_batch(data_loader, device, flatten)

    @abstractmethod
    def fit(self, data_loader, epochs, start_lr, device, model_path, weight_decay, gf=False, write_stats=False):
        pass

    def init_statistics(self):
        init_statistics(self.name)

    def write_statistics(self, writer, stat):
        write_statistics(writer, stat)


def init_statistics(name):
    statistics_path = f'statistics/{name}/'
    ew = open(f'{statistics_path}epoch_stat.csv', 'w')
    iw = open(f'{statistics_path}iteration_stat.csv', 'w')
    return ew, iw


def write_statistics(writer, stat_list):
    stat = '\n'.join(map(str, stat_list))
    writer.write(stat)
    stat_list.clear()
