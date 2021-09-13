import argparse


class ParameterHandler:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='train_script')

        # which model should be trained and how
        self.parser.add_argument('--models', type=str, default='convae',
                                 help='model to be trained. Available options: rotnet, simclr, convae.'
                                      'You can specify multiple models to be trained by giving their names'
                                      'separated by a comma, e.g. cifar,rotnet')
        self.parser.add_argument('--train_type', type=str, default='full',
                                 help='stage of training. Available options: pretrain, idec, full')

        # action params
        self.parser.add_argument('--train', type=str, default='True',
                                 help='if given, train the model')
        self.parser.add_argument('--evaluate', type=str, default='True',
                                 help='if True, compute KMeans, PCA, NMI and AC for the model and create visualizations')

        # path params
        self.parser.add_argument('--load_path', type=str, default=None,
                                 help='path to the pretrained model to be loaded, e.g. for further training or kmeans'
                                      'computation')
        self.parser.add_argument('--output_path', type=str, default=None,
                                 help='path to the dir where trained model and computed NMI, plots etc. should be'
                                      'stored')

        # training params
        self.parser.add_argument('--batch_size', type=int, default=128,
                                 help='batch size')
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate')
        self.parser.add_argument('--epochs', type=int, default=100,
                                 help='number of epochs')
        self.parser.add_argument('--degree_of_space_distortion', type=float, default=0.1,
                                 help='parameter controlling the DEC/IDEC\'s loss')

        # dataset params
        self.parser.add_argument('--datasets', type=str, default='',
                                 help='choose datasets which should be used. Available options: cifar, fmnist, stl10.'
                                      'Important! Give this param as a dataset, e.g. [cifar] or [cifar,stl10]'
                                      'Choose multiple datasets by giving them separated by a comma.')
        self.parser.add_argument('--data_path', type=str, default='./data',
                                 help='Path to the dataset(s), absolute or relative to the script\'s location')
        self.parser.add_argument('--download_data', type=str, default='True',
                                 help='Download the dataset(s)')
        self.parser.add_argument('--data_percent', type=float, default=1.0,
                                 help='Percent of data to be used for training/testing.')

        # SimCLR params
        self.parser.add_argument('--resnet', type=str, default='resnet18',
                                 help='default model used for SimCLR base')
        self.parser.add_argument('--tau', type=float, default=0.5,
                                 help='tau to be be used for SimCLR training')

        # KMeans params
        self.parser.add_argument('--n_clusters', type=int, default=10,
                                 help='Number of clusters for KMeans')

        self.args = self.parser.parse_args()

        self.args.datasets = self.args.datasets.split(',')
        self.args.models = self.args.models.split(',')
        
        self.args.train = 'True' == self.args.train
        self.args.evaluate = 'True' == self.args.evaluate
        self.args.download_data = 'True' == self.args.download_data

    def check_params(self):
        action_present = self.args.train or self.args.kmeans or self.args.pca

        if not action_present:
            raise ValueError('No action to be done. At least one of these parameters must be true:'
                             ' --train, --evaluate')

        if self.args.load_path is None:
            if self.args.train_type == 'idec':
                raise ValueError('Cannot train IDEC, no pretrained model provided at --load_path.')

            if not self.args.train:
                raise ValueError('Cannot perform kmeans or pca, no model provided at --load_path'
                                 'and no model can be trained because --train is set to False.')
