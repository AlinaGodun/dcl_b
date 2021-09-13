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
        self.parser.add_argument('--simclr_output_dim', type=int, default=128,
                                 help='Number of the output of the SimCLR\'s projection head')
        self.parser.add_argument('--simclr_resnet', type=str, default='resnet50',
                                 help='default model used for SimCLR base')
        self.parser.add_argument('--simclr_tau', type=float, default=0.5,
                                 help='tau to be be used for SimCLR training')

        # RotNet params
        self.parser.add_argument('--rotnet_num_classes', type=int, default=4,
                                         help='Number of classes the images should be classified into by RotNet')
        self.parser.add_argument('--rotnet_in_channels', type=int, default=3,
                                         help='Number of input channels for RotNet')
        self.parser.add_argument('--rotnet_num_blocks', type=int, default=3,
                                         help='Number of convolution blocks for RotNet')

        # ConvAE params
        self.parser.add_argument('--convae_n_channels', type=int, default=3,
                                         help='Number of input channels for ConvAE')
        self.parser.add_argument('--convae_n_classes', type=int, default=3,
                                         help='Number of classes for ConvAE. Must correspond to number of image '
                                              'channels')
        self.parser.add_argument('--convae_embd_sz', type=int, default=64,
                                         help='Size of the embedding space of ConvAE')

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

    def get_train_params(self, device, model_name):
        train_params = {
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.lr,
            'epochs': self.args.epochs,
            'device': device,
            'out_path': self.args.output_path
        }

        if 'DEC' in model_name and self.args.degree_of_space_distortion:
            train_params['degree_of_space_distortion'] = self.args.degree_of_space_distortion

        return train_params

    def get_dataset_params(self):
        dataset_params = {
            'train_path': self.args.data_path,
            'download': self.args.download_data,
            'data_percent': self.args.data_percent
        }
        return dataset_params

    def get_model_params(self, model_name):
        model_params = {}

        if model_name == 'simclr':
            model_params['output_dim'] = self.args.simclr_output_dim
            model_params['resnet_model'] = self.args.simclr_resnet
            model_params['tau'] = self.args.simclr_tau

        if model_name == 'convae':
            model_params['n_channels'] = self.args.convae_n_channels
            model_params['n_classes'] = self.args.convae_n_classes
            model_params['embd_sz'] = self.args.convae_embd_sz

        if model_name == 'rotnet':
            model_params['num_classes'] = self.args.rotnet_num_classes
            model_params['in_channels'] = self.args.rotnet_in_channels
            model_params['num_blocks'] = self.args.rotnet_num_blocks

        return model_params
