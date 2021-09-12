import os

import sklearn

from main.param_handler import ParameterHandler
from models.autoencoder.IDEC import IDEC
from models.autoencoder.conv_ae import ConvAE
from util.util import *

def train_model(model, batch_size, learning_rate, epochs, data, device, eval_data=None,
                degree_of_space_distortion=None):
    print(f"Training {model.name} started...")
    model = model.to(device)

    # paths to save/load models from
    base_path = "trained_models"
    model_name = f"{model.name}.pth"
    model_path = os.path.join(base_path, model_name)

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

    return model


def get_model(args, device):
    models = {
        'rotnet': RotNet,
        'simclr': SimCLR,
        'convae': ConvAE,
        'rotnet-idec': RotNetIDEC,
        'simclr-idec': SimClrIDEC,
        'convae-idec': IDEC
    }

    if args.load_path is None:
        model_name = args.parser.model_type
        if args.parser.train_type is 'idec':
            model_name = f'{model_name}-idec'
        # todo: add model's args
        model = models[model_name]()
    else:
        load_model(args.load_path, device)
    return model

def perform_action(model, args, device):
    if args.parser.train:
        train(model, args, device)

    if args.parser.evaluate:
        evaluate(model, args, device)

def train(model, args, device):
    train_params = {
        'model': model,
        'batch_size': args.parser.batch_size,
        'learning_rate': args.parser.lr,
        'epochs': args.parser.epochs,
        'device:': device
    }
    eval_models = ['cifar', 'cifar-idec', 'rotnet-idec']
    model_name = args.parser.model_type

    for dataset in args.parser.datasets:
        # TODO: add args for dataset
        train_params['data'] = model.get_dataset(dataset_name=dataset)

        if model_name in eval_models:
            # TODO: add args for dataset
            train_params['eval_data'] = model.get_dataset(eval_dataset=True)

        if 'idec' in model_name and args.parser.degree_of_space_distortion:
            train_params['degree_of_space_distortion'] = args.parser.degree_of_space_distortion

        model = train_model(**train_params)
    return

def evaluate(model, args, device):
    return

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

model = get_model(args, device)
perform_action(model, args, device)


