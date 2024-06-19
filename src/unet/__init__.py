
import glob
import json
import logging
import pandas as pd
from pathlib import Path
import torch

from . import data
from .model import load_model, get_model
from .evaluate import infere_single, get_model_name, get_model_config


def get_pretrained(
    model_path,
    channels,
    *,
    model_name: str = None,
    # network: str = None,
    device: torch.nn.Module = torch.device('cpu')
):
    # config
    model_path = Path(model_path)
    with open(model_path / model_name / 'config.json') as f:
        config = json.load(f)
    # model
    model = get_model(
        config['network'],
        in_channels=1,
        out_channels=1,
        channel=[0],
        drop_rate=0.,
    ).to(device)

    # load
    resume_model_file = model_path / model_name / 'model' / 'best_model.pt.tar'
    checkpoint = torch.load(resume_model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'model {model_name} loaded')
    print(f'model {model_name} loaded')
    return model


# def get_model_name(
#     # network: str = 'unet_2',
#     stego_method: str = 'LSBr',
#     # alpha: float = .4,
#     # loss: str = 'l1ws',
#     # drop_rate: float = .0,
#     # model_path: str = '../models/unet',
#     device: torch.device = torch.device('cpu')
# ) -> pd.DataFrame:
#     # list models
#     model_path = Path('../models/unet') / stego_method
#     models = glob.glob(str(model_path / '*' / 'config.json'))

#     # collect info
#     df = []
#     for model in map(Path, models):
#         # load config
#         model_name = model.parent.name
#         with open(model) as f:
#             config = json.load(f)

#         # load model
#         try:
#             model_file = model.parent / 'model' / 'best_model.pt.tar'
#             checkpoint = torch.load(model_file, map_location=device)
#         except FileNotFoundError:
#             logging.warning(f'no model found for {model_name}, skipped')
#             continue

#         if config.get('debug', False):
#             logging.warning(f'debug model {model_name} skipped')
#             continue
#         if config['alpha']:
#             config['alpha'] = float(config['alpha'])

#         # info
#         df.append({
#             # 'path': str(model.parent.parent),
#             'model_name': model_name,
#             'stego_method': config['stego_method'],
#             'alpha': config['alpha'],
#             'loss': config['loss'],
#             'network': config['network'],
#             'drop_rate': config['drop_rate'],
#             'epochs': checkpoint["epoch"],
#         })
#     df = pd.DataFrame(df)

#     # filter models
#     # df = df[df.network == network]
#     df = df[df.stego_method == stego_method]
#     # df = df[df.loss == loss]
#     # if alpha is not None:
#     #     df = df[df.alpha == alpha]
#     # else:
#     #     df = df[df.alpha.isna()]
#     # df = df[df.drop_rate == drop_rate]

#     #
#     if len(df) < 1:
#         raise RuntimeError(f'no model for {stego_method=} found')
#     if len(df) > 1:
#         raise RuntimeError(f'multiple models for {stego_method=} found')
#     return df['model_name'].iloc[0]


def get_unet_estimator(*args, **kw):
    # load model
    model = get_pretrained(*args, **kw)
    device = torch.device('cpu')

    def predict(x):
        y = infere_single(x, model=model, device=device)
        # print(x[1:4, 1:4, 0])
        # print(y[1:4, 1:4, 0])
        return y

    return predict
