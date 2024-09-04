"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import glob
import json
import logging
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import sys
import torch
import typing

sys.path.append('.')
import _defs
sys.path.append('unet')
# from data import get_data_loader, get_timm_transform
from data import get_timm_transform
from model import get_model
import fabrika

#
DEVICE = torch.device('cpu')
log = _defs.setup_custom_logger(pathlib.Path(__file__).name)


def infere_single(
    x: np.ndarray,
    model: typing.Callable,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    # convert to torch
    transform = get_timm_transform(
        mean=None,
        std=None,
        grayscale=True,
        demosaic_oracle=False,
        post_flip=False,
        post_rotate=False,
    )
    x_ = transform(x / 255.)[None].to(device)

    # infere
    y_ = model(x_)

    # convert back to numpy
    y = y_.detach().numpy()[0, 0, 1:-1, 1:-1] * 255.
    return y[..., None]


def get_model_name(
    stego_method: str = 'LSBR',
    model_dir: pathlib.Path = pathlib.Path('../models/unet'),
    device: torch.device = torch.device('cpu')
) -> pd.DataFrame:
    # list models
    model_path = model_dir / stego_method
    models = glob.glob(str(model_path / '*' / 'config.json'))

    # collect info
    df = []
    for model in map(pathlib.Path, models):
        # load config
        model_name = model.parent.name
        with open(model) as f:
            config = json.load(f)

        # load model
        try:
            model_file = model.parent / 'model' / 'best_model.pt.tar'
            checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        except FileNotFoundError:
            logging.warning(f'no model found for {model_name}, skipped')
            continue

        if config.get('debug', False):
            logging.warning(f'debug model {model_name} skipped')
            continue
        if config['alpha']:
            config['alpha'] = float(config['alpha'])

        # info
        df.append({
            'model_name': model_name,
            'stego_method': config['stego_method'],
            'alpha': config['alpha'],
            'loss': config['loss'],
            'network': config['network'],
            'drop_rate': config['drop_rate'],
            'epochs': checkpoint["epoch"],
        })
    #
    df = pd.DataFrame(df)
    df = df[df.stego_method == stego_method]

    #
    if len(df) < 1:
        raise RuntimeError(f'no model for {stego_method=} found')
    if len(df) > 1:
        raise RuntimeError(f'multiple models for {stego_method=} found')
    return df['model_name'].iloc[0]



def predict_unet(
    fname: str,
    model: torch.nn.Module,
    *,
    device: torch.nn.Module = torch.device('cpu'),
    imread: typing.Callable = _defs.imread4_f32,
    **kw,
):
    # load image
    x = imread(fname)
    x = x[..., 3:]

    # infere single image
    x_hat = infere_single(x, model=model, device=device)

    # difference image
    x = x[1:-1, 1:-1]

    # WS estimate
    x_bar = (x.astype('uint8') ^ 1).astype('float32')
    beta_hat = np.mean((x - x_bar) * (x - x_hat))

    # MAE estimate
    l1_hat = np.mean(np.abs(x - x_hat))

    #
    return {
        **kw,
        'beta_hat': beta_hat,
        'l1': l1_hat
    }


@fabrika.precovers(iterator='python', convert_to='pandas', ignore_missing=False, n_jobs=-1)  # n_jobs=os.cpu_count())
def predict_unet_cover(*args, **kw):
    return predict_unet(*args, **kw)


@fabrika.stego_spatial(iterator='python', convert_to='pandas', ignore_missing=False, n_jobs=-1)  # n_jobs=os.cpu_count())
def predict_unet_stego(*args, **kw):
    return predict_unet(*args, **kw)

def get_model_config(
    model_dir: pathlib.Path,
    stego_method: str,
    model_name: str,
) -> typing.Dict[str, typing.Any]:
    model_path = pathlib.Path(model_dir) / stego_method / model_name
    with open(model_path / 'config.json') as f:
        config = json.load(f)
    return config


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
    checkpoint = torch.load(resume_model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'model {model_name} loaded')
    return model

if __name__ == '__main__':

    #
    data_path = Path('../data')
    model_dir = pathlib.Path('../models/unet')
    stego_method = 'HILLR'  # dropout LSBR HILLR
    logging.basicConfig(level=logging.INFO)
    model_name = get_model_name(
        model_dir=model_dir,
        stego_method=stego_method,
        device=DEVICE,
    )
    
    model_name = get_model_name(
        stego_method=stego_method,
    )
    model = get_pretrained(
        model_path=model_dir / stego_method,
        channels=(3,),
        device=DEVICE,
        model_name=model_name,
    )

    df = predict_unet_cover(
        data_path,
        model=model,
        progress_on=True,
    )
    for sm in ['LSBR', 'HILLR']:
        df_stego = predict_unet_stego(
            data_path,
            model=model,
            stego_method=sm,
            progress_on=True,
        )        
        df = pd.concat([df, df_stego])

    outfile = f'../results/estimation/ws_{stego_method}.csv'
    df.to_csv(outfile, index=False)
    logging.info(f'output saved to {outfile}')
        