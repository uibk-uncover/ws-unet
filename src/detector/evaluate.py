"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

# import argparse
from glob import glob
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import timm
import torch
import typing

sys.path.append('.')
import _defs
sys.path.append('detector')
from data import get_timm_transform
from models import load_b0
import fabrika

#
DEVICE = torch.device('cpu')


def infere_single(
    x: np.ndarray,
    model: typing.Callable,
    lsbr_reference: bool = False,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    # convert to torch
    mean = list(timm.data.constants.IMAGENET_DEFAULT_MEAN)[1:2]
    std = list(timm.data.constants.IMAGENET_DEFAULT_STD)[1:2]
    transform = get_timm_transform(
        mean=mean,
        std=std,
        grayscale=True,
        demosaic_oracle=False,
        post_flip=False,
        post_rotate=False,
        lsbr_reference=lsbr_reference,
    )
    # print(transform)
    x_ = transform(x / 255.)[None].to(device)

    # infere
    with torch.no_grad():
        y_ = model(x_)
        y_pred = torch.nn.functional.softmax(y_, dim=1)

    # convert back to numpy
    y_pred = y_pred.detach().numpy()[0, 1]
    return y_pred


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

    # predict
    y = infere_single(x, lsbr_reference=lsbr_reference, model=model, device=device)
    # print(kw['name'], y, y > .5)

    return {
        **kw,
        'output': y,
        'prediction': y > .5
    }


def get_b0_detector(
    *args,
    lsbr_reference: bool = False,
    **kw,
):
    # load model
    device = torch.device('cpu')
    model = load_b0(*args, **kw, device=device)
    model.eval()

    def predict(x):
        y = infere_single(x, lsbr_reference=lsbr_reference, model=model, device=device)
        return y

    return predict


def get_model_name(
    # network: str = 'b0',
    stego_method: str = 'LSBR',
    alpha: float = .01,
    no_stem_stride: bool = False,
    lsbr_reference: bool = False,
    model_path: str = '../models/b0',
    device: torch.device = torch.device('cpu'),
) -> pd.DataFrame:
    # list models
    model_path = Path(model_path) / stego_method
    models = glob(str(model_path / '*' / 'config.json'))

    # collect info
    df = []
    for model in map(Path, models):
        # load config
        model_name = model.parent.name
        with open(model) as f:
            config = json.load(f)

        # load model
        try:
            model_file = model.parent / 'model' / 'best_model.pt.tar'
            checkpoint = torch.load(model_file, map_location=device)
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
            # 'path': str(model.parent.parent),
            'model_name': model_name,
            'stego_method': config['stego_method'],
            'alpha': config['alpha'],
            'loss': config['loss'],
            'network': config['network'],
            'drop_rate': config['drop_rate'],
            'lsbr_reference': config.get('lsbr_reference', False),
            'no_stem_stride': config.get('no_stem_stride', False),
            'epochs': checkpoint["epoch"],
        })
    df = pd.DataFrame(df)

    # filter models
    # df = df[df.network == network]
    df = df[df.stego_method == stego_method]
    # print(f'{alpha=}', df)
    df = df[df.alpha == alpha]
    # print(f'{no_stem_stride=}', df)
    df = df[df.no_stem_stride == no_stem_stride]
    # print(f'{lsbr_reference=}', df)
    df = df[df.lsbr_reference == lsbr_reference]

    #
    # print(df)
    if len(df) < 1:
        raise RuntimeError('no such model found')
    if len(df) > 1:
        raise RuntimeError('multiple such models found')
    return df['model_name'].iloc[0]


@fabrika.precovers(iterator='python', convert_to='pandas', ignore_missing=False, n_jobs=-1)  # n_jobs=os.cpu_count())
def predict_b0_cover(*args, **kw):
    return predict_unet(*args, **kw)


@fabrika.stego_spatial(iterator='python', convert_to='pandas', ignore_missing=False, n_jobs=-1)  # n_jobs=os.cpu_count())
def predict_b0_stego(*args, **kw):
    return predict_unet(*args, **kw)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #
    stego_method = 'LSBR'
    data_path = Path('../data')
    model_dir = Path('../models/b0') / stego_method
    no_stem_stride = False
    lsbr_reference = False

    model_name = get_model_name(
        no_stem_stride=no_stem_stride,
        lsbr_reference=lsbr_reference,
    )

    # load model
    in_channels = 2 if lsbr_reference else 1
    model = load_b0(
        model_dir=model_dir,
        model_name=model_name,
        in_channels=in_channels,
        shape=(512, 512),
        device=DEVICE,
        # no_stem_stride=no_stem_stride,
        # lsbr_reference=lsbr_reference,
    )
    model.eval()

    df = predict_b0_cover(data_path, model=model, device=DEVICE)
    for sm in ['LSBR', 'HILLR']:
        df_stego = predict_b0_stego(data_path, model=model, stego_method=sm, device=DEVICE)
        df = pd.concat([df, df_stego])

    outfile = f'../results/detection/b0.csv'
    df.to_csv(outfile, index=False)
    logging.info(f'output saved to {outfile}')
        
