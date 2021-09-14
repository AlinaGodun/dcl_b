import os

import sklearn

from main.param_handler import ParameterHandler
from util import util
from util.cluster_accuracy import cluster_accuracy
from util.util import *

def train_model(model, batch_size, learning_rate, epochs, data, device, eval_data=None,
                degree_of_space_distortion=None, out_path='trained_models'):
    print(f"Training {model.name} started...")
    model = model.to(device)

    # paths to save/load models from
    model_name = f"{model.name}.pth"
    model_path = os.path.join(out_path, model_name)

    print(f"Number of train images: {len(data)}")

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    if eval_data is not None:
        eval_loader = torch.utils.data.DataLoader(eval_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

    train_params = {
        'data_loader': train_loader,
        'epochs': epochs,
        'start_lr': learning_rate,
        'device': device,
        'model_path': model_path
    }

    if degree_of_space_distortion:
        train_params['degree_of_space_distortion'] = degree_of_space_distortion

    if eval_data:
        train_params['eval_data_loader'] = eval_loader

    model = model.fit(**train_params)

    model.eval()
    torch.save(model.state_dict(), model_path)
    print(f'Saved trained {model.name} at: {model_path}')

    return model


def create_base_model(model_name, param_handler, device):
    print('_______________________________')
    available_models = {
        'rotnet': RotNet,
        'simclr': SimCLR,
        'convae': ConvAE
    }

    if model_name not in available_models.keys():
        raise KeyError(f'No model {model_name} is available. Available models are: rotnet, simclr, cifar.')

    model_params = param_handler.get_model_params(model_name)
    print(f'Creating model: {model_name} with params: {model_params}')

    model = available_models[model_name](**model_params)
    model = model.to(device)

    return model

def create_idec_model(model, model_name, param_handler, device, dataset_name):
    print('_______________________________')
    available_models = {
        'rotnet-idec': RotNetIDEC,
        'simclr-idec': SimClrIDEC,
        'convae-idec': IDEC
    }

    if 'idec' not in model_name:
        model_name = f'{model_name}-idec'

    if model_name not in available_models.keys():
        raise KeyError(f'No model {model_name} is available. Available models are: rotnet, simclr, cifar.')

    #
    dataset_params = param_handler.get_dataset_params('simclr')
    dataset_params['transforms'] = False

    cluster_data = model.get_dataset(dataset_name, dataset_params)
    cluster_loader = torch.utils.data.DataLoader(cluster_data,
                                               batch_size=param_handler.args.batch_size,
                                               shuffle=True,
                                               drop_last=True)

    print(f'Creating model: {model_name} with base model: {model.name}')
    model = available_models[model_name](model=model, train_loader=cluster_loader)
    model = model.to(device)

    return model

def load_model(args, device):
    print('_______________________________')
    print(f'Loading existing model at path: {args.load_path}')
    model = util.load_model(args.load_path, device)
    return model

def perform_action(model, model_name, param_handler, device, dataset):
    if param_handler.args.train_type == 'pretrain':
        result_model = perform_model_action(model, model_name, param_handler, device, dataset)

    if param_handler.args.train_type == 'idec':
        idec = create_idec_model(model, model_name, param_handler, device, dataset)
        result_model = perform_model_action(idec, model_name, param_handler, device, dataset)

    if param_handler.args.train_type == 'full':
        result_model = perform_model_action(model, model_name, param_handler, device, dataset)
        idec = create_idec_model(result_model, model_name, param_handler, device, dataset)

        result_model = perform_model_action(idec, model_name, param_handler, device, dataset)

    return result_model

def perform_model_action(model, model_name, param_handler, device, dataset):
    if param_handler.args.train:
        print('_______________________________')
        print(f'Started {model.name} training...')
        model = train(model, param_handler, device, dataset)

    if param_handler.args.evaluate:
        print('_______________________________')
        print(f'Started {model.name} evaluation...')
        evaluate(model, model_name, param_handler, device, dataset)

    return model

def train(model, param_handler, device, dataset):
    eval_models = ['IDEC_RotNet', 'AE', 'IDEC_AE']
    model_name = model.name

    train_params = param_handler.get_train_params(device, model_name)
    dataset_params = param_handler.get_dataset_params(model_name)

    print(f'Training parameters: {train_params}')
    print(f'Dataset parameters: {dataset_params}')

    print('_______________________________')
    print(f'Training dataset: {dataset}')
    train_params['data'] = model.get_dataset(dataset, dataset_params)

    if model_name in eval_models:
        dataset_params['eval_dataset'] = True
        train_params['eval_data'] = model.get_dataset(dataset, dataset_params)

    if dataset.upper() not in model.name:
        model.name = f'{model.name}_{dataset.upper()}'
    model = train_model(model=model, **train_params)

    return model

def evaluate(model, model_name, param_handler, device, dataset):
    evaluation_params = {
        'model': model,
        'name': model.name,
        'device': device
    }

    dataset_params = param_handler.get_dataset_params(model_name)
    dataset_params['train'] = False

    if 'RotNet' in model.name:
        evaluation_params['flatten'] = True
        evaluation_params['layer'] = 'conv2'

    if args.n_clusters:
        evaluation_params['n_clusters'] = args.n_clusters

    d = model.get_dataset(dataset, dataset_params)
    evaluation_params['data_loader'] = torch.utils.data.DataLoader(d, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True)
    evaluation_params['colors_classes'] = {i: color_class for i, color_class in zip(range(len(d.classes)), d.classes)}
    labels, kmeans, nmi, reduced_data, lable_classes = compute_nmi_and_pca(**evaluation_params)
    ca = cluster_accuracy(labels, kmeans.labels_)
    print(f'Kmeans NMI: {nmi:.3f}')
    print(f'Cluster Accuracy: {ca:.3f}')

    out_path = os.path.join(param_handler.args.output_path, model.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print(f'Saving evaluation data and visualizations at: {out_path}')
    plot_pca_and_nmi(model.name, nmi, reduced_data, lable_classes, out_path=out_path)
    plot_class_representation(reduced_data, model.name, lable_classes)


print("Versions:")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {np.__version__}", )
print(f"scikit-learn: {sklearn.__version__}")

device = detect_device()
print("Using device: ", device)

param_handler = ParameterHandler()
param_handler.check_params()
args = param_handler.args

for dataset in args.datasets:
    if args.load_path:
        model = load_model(args, device)
        model_name = args.models[0]
        model = perform_action(model, model_name, param_handler, device, dataset)
    else:
        for model_name in args.models:
            model = create_base_model(model_name, param_handler, device)
            model = perform_action(model, model_name, param_handler, device, dataset)


