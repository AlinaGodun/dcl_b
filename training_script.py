import os

import sklearn

from main.param_handler import ParameterHandler
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


def create_models(args, device):
    print('_______________________________')
    available_models = {
        'rotnet': RotNet,
        'simclr': SimCLR,
        'convae': ConvAE,
        'rotnet-idec': RotNetIDEC,
        'simclr-idec': SimClrIDEC,
        'convae-idec': IDEC
    }

    models = []

    for model_name in args.models:
        if args.train_type == 'idec':
            model_name = f'{model_name}-idec'
        # todo: add model's args
        print(f'Creating model: {model_name}')
        models.append(available_models[model_name]())

    return models

def load_model(args, device):
    print('_______________________________')
    print(f'Loading existing model at path: {args.load_path}')
    load_model(args.load_path, device)
    return model

def perform_action(model, args, device):
    if args.train:
        print('_______________________________')
        print('Started model training...')
        model = train(model, args, device)

    if args.evaluate:
        print('_______________________________')
        print('Started model evaluation...')
        evaluate(model, args, device)

def train(model, args, device):
    eval_models = ['cifar', 'cifar-idec', 'rotnet-idec']
    model_name = model.name

    train_params = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'device': device,
        'out_path': args.output_path
    }

    dataset_params = {
        'train_path': args.data_path,
        'download': args.download_data,
        'data_percent': args.data_percent
    }

    if 'DEC' in model_name and args.degree_of_space_distortion:
        train_params['degree_of_space_distortion'] = args.degree_of_space_distortion

    print(f'Training parameters: {train_params}')
    print(f'Dataset parameters: {dataset_params}')

    for dataset in args.datasets:
        print('_______________________________')
        print(f'Training dataset: {dataset}')
        train_params['data'] = model.get_dataset(dataset_name=dataset, **dataset_params)

        if model_name in eval_models:
            train_params['eval_data'] = model.get_dataset(eval_dataset=True, **dataset_params)

        model = train_model(model=model, **train_params)
    return model

def evaluate(model, args, device):
    evaluation_params = {
        'model:': model,
        'name:': model.name,
        'device': device
    }
    if 'RotNet' in model.name:
        evaluation_params['flatten'] = True
        evaluation_params['layer'] = 'conv2'

    if args.n_clusters:
        evaluation_params['n_clusters'] = args.n_clusters

    for dataset in args.datasets:
        # TODO: add args for dataset
        d = model.get_dataset(dataset_name=dataset, train=False)
        evaluation_params['data_loader'] = torch.utils.data.DataLoader(d, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=True)
        evaluation_params['colors_classes'] = {i: color_class for i, color_class in zip(range(len(d.classes)), d.classes)}
        labels, kmeans, nmi, reduced_data, lable_classes = compute_nmi_and_pca(**evaluation_params)


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

if args.load_path:
    model = load_model(args, device)
else:
    models = create_models(args, device)
    for model in models:
        perform_action(model, args, device)


