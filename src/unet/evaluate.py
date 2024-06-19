"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import collections
import glob
import json
import logging
import numpy as np
import pandas as pd
import pathlib
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import torchvision.transforms as transforms
import typing

sys.path.append('.')
sys.path.append('unet')
from data import get_data_loader, get_timm_transform
from model import get_model
sys.path.append('..')
from _defs import seed_everything, setup_custom_logger
from _defs import metrics, losses

#
DEVICE = torch.device('cpu')
log = setup_custom_logger(pathlib.Path(__file__).name)


def evaluate_model(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion, device,
    loss_meter,
    mask_meters,
    target_meters,
) -> float:

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)
            targets[0] = targets[0].to(device, non_blocking=True)
            alphas = targets[1].numpy()
            targets[1] = targets[1].to(device, non_blocking=True)
            covers = targets[0]

            # Forward-propagate
            outputs = model(images)

            # Collect input dropout mask
            dropout_mask = None

            # Compute loss
            loss = criterion(outputs, targets, images)

            # Record performance
            images = images.cpu().numpy()
            covers = covers.cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            loss_meter.update(loss.item(), batch_size)
            for meter in mask_meters:
                meter.update(
                    covers,
                    outputs,
                    dropout_mask,
                )
            for meter in target_meters:
                meter.update(
                    images,
                    outputs,
                    alphas,
                )


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


def evaluate_for_dataset(loader, model, criterion, device, suffix):

    # Create meters
    loss_meter = metrics.AverageMeter('Loss', ':.4e')
    mask_meters = [
        metrics.MAEMeter('MAE', ':.4e')
    ]
    target_meters = [
        metrics.WSMeter('WS', ':.4e'),
    ]

    # Evaluate model
    evaluate_model(
        loader=loader,
        model=model,
        criterion=criterion,
        device=device,
        loss_meter=loss_meter,
        mask_meters=mask_meters,
        target_meters=target_meters,
    )

    # Write predictions
    return {
        'loss': loss_meter.avg,
        **({
            v.name: v.avg
            for v in (mask_meters + target_meters)
        }),
    }


def evaluate(
    model_name: str,
    split: str,
    config: typing.Dict[str, typing.Any],
):
    # Set up directory name for output directory
    output_path = config['output_dir'] / pathlib.Path(model_name)
    output_path.mkdir(exist_ok=True)
    model_path = config['model_path']
    result_file = output_path / 'results.csv'

    # Get model and log directories
    # Concatenate path to one subdirectory for logging
    log_dir = output_path / 'log'
    model_file = model_path / 'model' / 'best_model.pt.tar'

    # Decide whether to run on GPU or CPU
    if torch.cuda.is_available():
        log.info('Using GPU')
        device = torch.device('cuda')
    else:
        log.info('Using CPU, this will be slow')
        device = torch.device('cpu')

    # Seed if requested
    if config['seed']:
        log.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        seed_everything(config['seed'])

    # Data loaders
    config['pre_rotate'] = config['post_rotate'] = config['post_flip'] = config['pre_flip'] = False
    te_loader, te_dataset = get_data_loader(split, config)
    print(f'Evaluating on {len(te_loader)} batches')

    # Input channels
    in_channels = 1 if config['grayscale'] else 3
    in_channels += 1 if config['parity_oracle'] else 0
    in_channels += 3 if config['demosaic_oracle'] else 0
    out_channels = 1

    # Set up model
    model = get_model(
        config['network'],
        in_channels=in_channels,
        out_channels=out_channels,
        channel=config['channel'],
        drop_rate=0.0,
    ).to(device)
    summary(
        model,
        input_size=(config['batch_size'], in_channels, *config['shape'])
    )

    # Set up loss and optimizer
    if config['loss'] == 'l1':
        criterion = losses.L1Loss().to(device)
    elif config['loss'] == 'l1ws':
        criterion = losses.L1WSLoss().to(device)
    else:
        raise NotImplementedError(f'loss {config["loss"]} not implemented')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    log.info(f'=> loaded trained model {model_file} ({checkpoint["epoch"]} epochs)')

    # Evaluation
    scores = evaluate_for_dataset(
        loader=te_loader,
        model=model,
        criterion=criterion,
        device=device,
        suffix='',
    )

    # write columns (on first write)
    config_cols = ['stego_method', 'dataset', 'split']
    score_cols = ['loss', 'mae', 'ws']
    if not pathlib.Path(result_file).exists():
        with open(result_file, 'a') as f:
            f.write(
                ','.join(config_cols + score_cols) + '\n'
            )
    # write results
    print(scores)
    with open(result_file, 'a') as f:
        s = [
            str(config[c])
            for c in config_cols
        ] + [
            str(scores[c])
            for c in score_cols
        ]
        f.write(','.join(s) + '\n')


def get_model_name(
    stego_method: str = 'LSBr',
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


def get_model_config(
    model_dir: pathlib.Path,
    stego_method: str,
    model_name: str,
) -> typing.Dict[str, typing.Any]:
    model_path = pathlib.Path(model_dir) / stego_method / model_name
    with open(model_path / 'config.json') as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    #
    model_dir = pathlib.Path('../models/unet')
    stego_method = 'dropout'  # dropout LSBR HILLR
    model_name = get_model_name(
        model_dir=model_dir,
        stego_method=stego_method,
        device=DEVICE,
    )

    #
    config = get_model_config(
        model_dir=model_dir,
        stego_method=stego_method,
        model_name=model_name,
    )
    config['model_path'] = model_dir / stego_method / model_name
    config['output_dir'] = pathlib.Path('../results/prediction/')
    config['dataset'] = pathlib.Path('../data/')
    config['batch_size'] = 1
    config['print_freq'] = 1

    #
    for split in ['split_tr.csv', 'split_va.csv', 'split_te.csv']:
        config['split'] = split
        evaluate(
            model_name=model_name,
            split=split,
            config=config,
        )
