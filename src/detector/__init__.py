

import glob
import json
import logging
import pandas as pd
from pathlib import Path
import torch

from .models import load_b0
from .evaluate import infere_single


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
    alpha: float = .4,
    no_stem_stride: bool = False,
    lsbr_reference: bool = False,
    model_path: str = '../models/b0',
    device: torch.device = torch.device('cpu'),
) -> pd.DataFrame:
    # list models
    model_path = Path(model_path) / stego_method
    models = glob.glob(str(model_path / '*' / 'config.json'))

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
    print(f'{alpha=}', df)
    df = df[df.alpha == alpha]
    print(f'{no_stem_stride=}', df)
    df = df[df.no_stem_stride == no_stem_stride]
    print(f'{lsbr_reference=}', df)
    df = df[df.lsbr_reference == lsbr_reference]

    #
    print(df)
    if len(df) < 1:
        raise RuntimeError('no such model found')
    if len(df) > 1:
        raise RuntimeError('multiple such models found')
    return df['model_name'].iloc[0]
