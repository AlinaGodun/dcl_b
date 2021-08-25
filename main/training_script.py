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

    if degree_of_space_distortion is None:
        model = model.fit(data_loader=train_loader, epochs=epochs, start_lr=learning_rate, device=device,
                          model_path=model_path)
    else:
        if eval_data is None:
            model = model.fit(data_loader=train_loader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=model_path,
                              degree_of_space_distortion=degree_of_space_distortion)
        else:
            model = model.fit(data_loader=train_loader, epochs=epochs, start_lr=learning_rate, device=device,
                              model_path=model_path,
                              degree_of_space_distortion=degree_of_space_distortion,
                              eval_data_loader=eval_loader)

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



