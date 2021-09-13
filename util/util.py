import torch
import torchvision
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from models.autoencoder.IDEC import IDEC
from models.autoencoder.conv_ae import ConvAE
from models.simclr.IDEC import IDEC as SimClrIDEC
from models.rotnet.DEC import DEC as RotNetIDEC
from models.rotnet.rotnet import RotNet
from models.simclr.simclr import SimCLR
import seaborn as sns
from scipy import spatial


def plot_images(images: torch.Tensor, pad: int = 0, nrow: int=8):
    """Aligns multiple images on an N by 8 grid"""
    def imshow(img):
        plt.figure(figsize=(10, 22))
        npimg = img.numpy()
        npimg = np.array(npimg)
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1)
    imshow(torchvision.utils.make_grid(images, pad_value=255, normalize=False, padding=pad, nrow=nrow))
    plt.show()


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def load_model(name, device, cluster_centres=None, model_params={}):
    """
        Utility method for downloading the models by name. The model should be located in trained_models
        folder.
    Parameters:
                name (str): name of the model. The name must either contain RotNet, CIFAR or AE. If it's a (I)DEC version,
                it must contain DEC in the name. If it is SimCLR with resnet18 base, it must contain r18 in the name.
                device (str): device where model should be loaded
                cluster_centres (torch.Tensor): structure to contain (I)DEC's cluster module. Must have correct
                dimensionality. Only relevant for IDEC models. Default values provided for RotNet and SimCLR, use if
                dimensionality of your model is not default one
            Returns:
                model (IDEC): trained model
    """
    if 'RotNet' in name:
        if cluster_centres is None:
            cluster_centres = torch.rand(size=(10, 12288))

        if 'DEC' not in name:
            model = RotNet(num_classes=4)
        else:
            model = RotNetIDEC(model=RotNet(num_classes=4), cluster_centres=cluster_centres, device=device)
    elif 'SimCLR' in name:
        if 'r18' in name:
            resnet_model = 'resnet18'
        else:
            resnet_model = 'resnet50'

        if 'DEC' not in name:
            model = SimCLR(resnet_model=resnet_model)
        else:
            if cluster_centres is None:
                cluster_centres = torch.rand(size=(10, 2048))
            model = SimClrIDEC(cluster_centres=cluster_centres)
    elif 'AE' in name:
        if 'DEC' not in name:
            model = ConvAE(n_channels=3, n_classes=3, embd_sz=64)
        else:
            model = IDEC()

    state_dict = torch.load(f'trained_models/{name}', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def compute_nmi_and_pca(model, name, colors_classes, device, data_loader, flatten=True, layer='conv2', n_clusters=None):
    """
    Utility method for computing K-Means and PCA.
    Parameters:
                model (AbstractModel or AbstractDecModel): model which should be used to encode the data
                name (str): name of the model. The name must either contain RotNet, CIFAR or AE. If it's a (I)DEC version,
                it must contain DEC in the name. If it is SimCLR with resnet18 base, it must contain r18 in the name.
                colors_classes (dictionary): dict containing mapping from names to labels, e.g. {1: 'airplane', ...}.
                device (str): device to be used for data encoding
                data_loader (DataLoader): dataloader containing data to be used for K-Means and PCA
                flatten (Boolean): if flattening of output is needed; only used for RotNet models
                layer (str): output of which layer should be used for encoding. only used for RotNet models
                n_clusters (int): number of clusters to be used for K-Means. By default is set to the number of unique
                labels in the dataset
            Returns:
                labels: ground-truth labels
                kmeans: KMeans object containing results of the performed clustering
                nmi: NMI between ground-truth and kmeans labels
                reduced_data: data reduced with PCA (number of components = 2)
                lable_classes: array containing ground_truth labels, but in text format instead of int. To be used
                when plotting different clusters
    """
    model.eval()
    if 'pretrained' in name:
        decoder = model
    else:
        decoder = model.model

    if 'RotNet' in name:
        embedded_data, labels = decoder.forward_batch(data_loader, device=device, flatten=flatten, layer=layer)
    else:
        embedded_data, labels = decoder.forward_batch(data_loader, device)
    lable_classes = [colors_classes[l] for l in labels]

    if n_clusters is None:
        n_clusters = len(set(labels))
    print(f'Starting K-Means with {n_clusters} clusters...')
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)

    print('Starting PCA with 2 components...')
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embedded_data)

    return labels, kmeans, nmi, reduced_data, lable_classes


def compute_nmi_and_pca_for_plot(model, name, colors_classes, device, testloader, flatten=True, layer='conv2'):
    if 'pretrained' in name:
        decoder = model
    else:
        decoder = model.model

    if 'RotNet' in name:
        embedded_data, labels, aug_labels = decoder.forward_batch(testloader, device=device, flatten=flatten, layer=layer)
    else:
        embedded_data, labels, aug_labels = decoder.forward_batch(testloader, device)
    lable_classes = [colors_classes[l] for l in labels]

    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)

    embedded_data = np.concatenate((embedded_data, kmeans.cluster_centers_))
    labels = np.concatenate((labels, np.array(list(range(0, 10)))))
    aug_labels = np.concatenate((aug_labels, np.array([-1] * 10)))
    lable_classes += [-1] * 10

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embedded_data)

    return labels, aug_labels, kmeans, nmi, reduced_data, lable_classes


def plot_pca_and_nmi(name, nmi, pca, lable_classes, axes=None):
    """
    Create plot of the reduced data and add model's name and NMI to the title
    Parameters:
                name (str): models' name
                nmi (float): NMI to be added to the title
                pca (list): data reduced with pca
                lable_classes (list): structure containing labels of each entry in the reduced_data. Needed to color
                the points belonging to the same class into different colors
                axes (matplotlib axes): axes where plot should be located. If none, create own plot with size (8,8)
    """
    if axes is None:
        f, axes = plt.subplots(figsize=(8, 8))
    axes.set_title(f'{name} Kmeans NMI: {nmi:.4f}')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.axis('off')
    sns.scatterplot(ax=axes, x=pca[:,0], y=pca[:,1], hue=lable_classes, s=7, palette='viridis')


def plot_class_representation(pca, name, lable_classes, aug_labels=None):
    """
    Create multiple plots for each of the ground-truth classes and plot them together in a grid
    Parameters:
                name (str): models' name
                pca (list): data reduced with pca
                lable_classes (list): structure containing labels of each entry in the reduced_data. Needed to color
                the points belonging to the same class into different colors
                axes (matplotlib axes): axes where plot should be located. If none, create own plot with size (8,8)
    """
    print(f'{name} class representation')
    lc = np.array(lable_classes)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    normal_points = pca[:-10]
    centers_coordinates = pca[-10:]
    tree = spatial.KDTree(normal_points)
    nearest_ids = []

    s = set(lable_classes)
    s.remove(-1)

    for i, c in enumerate(s):
        nearest_ids.append(tree.query(centers_coordinates[i], k=5)[1])

    for i, c in enumerate(s):
        class_labels = lc == c
        originals = aug_labels == 1
        augmented = aug_labels == 0

        ids_original = np.where(np.logical_and(class_labels, originals))[0]
        ids_augmented = np.where(np.logical_and(class_labels, augmented))[0]

        axes[i].set(title=f'class {c}')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].axis('off')
        sns.scatterplot(ax=axes[i], x=pca[:, 0], y=pca[:, 1], s=7, color='#d1dade')
        sns.scatterplot(ax=axes[i], x=pca[ids_augmented, 0], y=pca[ids_augmented, 1], s=10, alpha=0.5)
        sns.scatterplot(ax=axes[i], x=pca[ids_original, 0], y=pca[ids_original, 1], s=10, color='#ff802b', alpha=0.5)

        for k in range(10):
            # sns.scatterplot(ax=axes[i], x=pca[nearest_ids[k], 0], y=pca[nearest_ids[k], 1], s=50, color='#000000', marker='X')
            axes[i].annotate(str(k), xy=(pca[nearest_ids[k]][0][0], pca[nearest_ids[k]][0][1]))
        # sns.scatterplot(ax=axes[i], x=pca[centres, 0], y=pca[centres, 1], s=10, color='#000000', marker='s')
        # sns.scatterplot(ax=axes[i], x=pca[nearest_ids, 0], y=pca[nearest_ids, 1], s=50, color='#000000', marker='X')

    plt.show()

    return nearest_ids

def plot_class_representation_with_centers(pca, name, lable_classes, aug_labels, centers):
    print(f'{name} class representation')
    lc = np.array(lable_classes)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    normal_points = pca[:-10]
    centers = pca[-10]
    tree = spatial.KDTree(normal_points)

    s = set(lable_classes)

    for i, c in enumerate(set(lable_classes)):
        class_labels = lc == c
        originals = aug_labels == 1
        augmented = aug_labels == 0

        nearest_ids = tree.query(centers[c], k=5)[1]

        ids_original = np.where(np.logical_and(class_labels, originals))[0]
        ids_augmented = np.where(np.logical_and(class_labels, augmented))[0]

        axes[i].set(title=f'class {c}')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].axis('off')
        sns.scatterplot(ax=axes[i], x=pca[:, 0], y=pca[:, 1], s=7, color='#d1dade')
        sns.scatterplot(ax=axes[i], x=pca[ids_augmented, 0], y=pca[ids_augmented, 1], s=10, alpha=0.5)
        sns.scatterplot(ax=axes[i], x=pca[ids_original, 0], y=pca[ids_original, 1], s=10, color='#ff802b',
                        alpha=0.5)
        sns.scatterplot(ax=axes[i], x=pca[ids_original, 0], y=pca[ids_original, 1], s=10, color='red',
                        alpha=0.5)
        sns.scatterplot(ax=axes[i], x=pca[ids_original, 0], y=pca[ids_original, 1], s=10, color='red',
                        alpha=0.5)
        sns.scatterplot(ax=axes[i], x=pca[nearest_ids, 0], y=pca[nearest_ids, 1], s=10, color='red',
                        alpha=0.5)

    plt.show()


def plot_class_representation_rotnet(pca, name, lable_classes, aug_labels):
    print(f'{name} class representation')
    lc = np.array(lable_classes)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, c in enumerate(set(lable_classes)):
        class_labels = lc == c

        ids = np.where(class_labels)[0]
        hue = aug_labels[ids]

        axes[i].set(title=f'class {c}')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].axis('off')
        sns.scatterplot(ax=axes[i], x=pca[:, 0], y=pca[:, 1], s=7, color='#d1dade')
        sns.scatterplot(ax=axes[i], x=pca[ids, 0], y=pca[ids, 1], s=10, hue=hue,
                        alpha=0.5, palette='viridis')

    plt.show()
