import torch
import torchvision
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from models.simclr.IDEC import IDEC as SimClrIDEC
from models.rotnet.IDEC import IDEC as RotNetIDEC
from models.rotnet.rotnet import RotNet
from models.simclr.simclr import SimCLR
import seaborn as sns


def denormalize(tensor: torch.Tensor, mean: float = 0.1307, std: float = 0.3081) -> torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize the mnist images properly.
    """
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, 3, 32, 32) * 255).int().detach()


def plot_images(images: torch.Tensor, pad: int = 0, nrow: int=8):
    """Aligns multiple images on an N by 8 grid"""
    def imshow(img):
        plt.figure(figsize=(10, 22))
        npimg = img.numpy()
        npimg = np.array(npimg)
        plt.axis('off')
        # plt.imshow(np.transpose(npimg, (1, 2, 0)),
        #            vmin=0, vmax=1)
        plt.imshow(np.transpose(npimg, (1,2,0)), vmin=0, vmax=1)
    imshow(torchvision.utils.make_grid(images, pad_value=255, normalize=False, padding=pad, nrow=nrow))
    plt.show()


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    labels = []
    for batch, blabels in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
        labels = labels + blabels.tolist()
    return torch.cat(embeddings, dim=0).numpy(), labels


def decode_batchwise(dataloader, model, device):
    """ Utility function for decoding the whole data set in a mini-batch fashion
    """
    decodings = []
    for batch in dataloader:
        batch_data = batch[0].to(device)
        decodings.append(model(batch_data).detach().cpu())
    return torch.cat(decodings, dim=0).numpy()


def predict_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[0].to(device)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def evaluate_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for evaluating the cluster performance with NMI in a mini-batch fashion
    """
    predictions = []
    labels = []
    for batch in dataloader:
        batch_data = batch[0].to(device)
        label = batch[1]
        labels.append(label)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return normalized_mutual_info_score(labels, predictions)


def load_model(name, device, cluster_centres=torch.rand(size=(10, 12288))):
    if 'RotNet' in name:
        if 'DEC' not in name:
            model = RotNet(num_classes=4)
        else:
            model = RotNetIDEC(model=RotNet(num_classes=4), cluster_centres=cluster_centres, device=device)
    elif 'SimCLR' in name:
        if 'r18' in name:
            resnet_model = 'resnet18'
        else:
            resnet_model = 'resnet50'

        if 'pretrained' in name:
            model = SimCLR(resnet_model=resnet_model)
        else:
            if ('c10') in name:
                model = SimClrIDEC(cluster_centers=torch.rand(size=(10, 2048)))
            elif 'c30' in name:
                model = SimClrIDEC(cluster_centers=torch.rand(size=(30, 2048)))
            else:
                model = SimClrIDEC()

    state_dict = torch.load(f'trained_models/{name}', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def compute_nmi_and_pca(model, name, colors_classes, device, testloader, flatten=True):
    if 'pretrained' in name:
        decoder = model
    else:
        decoder = model.model

    if 'RotNet' in name:
        embedded_data, labels = decoder.forward_batch(testloader, device=device, flatten=flatten)
    else:
        embedded_data, labels = decoder.forward_batch(testloader, device)
    lable_classes = [colors_classes[l] for l in labels]

    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embedded_data)

    return labels, kmeans, nmi, reduced_data, lable_classes


def plot_pca_and_nmi(name, axes, nmi, pca, lable_classes):
    axes.set_title(f'{name} Kmeans NMI: {nmi:.4f}')
    sns.scatterplot(ax=axes, x=pca[:,0], y=pca[:,1], hue=lable_classes, s=7, palette='viridis')


def plot_class_representation(pca, name, lable_classes):
    print(f'{name} class representation')
    lc = np.array(lable_classes)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, c in enumerate(set(lable_classes)):
        ids = np.where(lc == c)[0]
        axes[i].set(title=f'class {c}')
        axes[i].tick_params(bottom=False, left=False)
        axes[i].set(xticklabels=[], yticklabels=[])
        sns.scatterplot(ax=axes[i], x=pca[:, 0], y=pca[:, 1], s=7, color='#d1dade')
        sns.scatterplot(ax=axes[i], x=pca[ids, 0], y=pca[ids, 1], s=7, color='#ff802b')

    plt.show()
