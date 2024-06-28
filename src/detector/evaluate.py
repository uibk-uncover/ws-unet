"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import json
import numpy as np
import pathlib
import sys
import timm
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import torchvision.transforms as transforms
import typing

sys.path.append('.')
import _defs
sys.path.append('detector')
from data import get_data_loader, get_timm_transform
from models import get_b0

log = _defs.setup_custom_logger(pathlib.Path(__file__).name)

ARGS_COLS = [
    'model',
    'dataset',
    'te_csv',
]
SCORES_COLS = [
    'loss',
    'accuracy',
    'misclassification',
    'precision',
    'recall',
    'p_e',
    'p_md^5fp',
    'wauc',
]


def evaluate_model(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion, device,
    loss_meter,
    target_meters,
    score_meters,
) -> float:

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Compute output
            logits = model(images)
            outputs = torch.nn.functional.softmax(logits, dim=1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Skip softmax activation as we are only interested in the argmax
            _, predictions = torch.max(outputs, dim=1)

            # Record performance
            targets = targets.cpu().numpy()
            outputs = outputs[:, 1].detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            loss_meter.update(loss.item(), batch_size)
            for meter in score_meters:
                meter.update(
                    targets,
                    outputs,
                )
            for meter in target_meters:
                meter.update(
                    targets,
                    predictions,
                )


def infere_single(
    x: np.ndarray,
    model: typing.Callable,
    lsbr_reference: bool = False,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    # convert to torch
    # print(x.shape, x[:3, :3, 0])
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


def evaluate_for_dataset(loader, model, criterion, device, suffix=''):

    # Create meters
    loss_meter = _defs.metrics.LossMeter()
    target_meters = [
        _defs.metrics.AccuracyMeter(),
        _defs.metrics.MisclassificationMeter(),
        _defs.metrics.PrecisionMeter(),
        _defs.metrics.RecallMeter(),
    ]
    score_meters = [
        _defs.metrics.PEMeter(),
        _defs.metrics.PMD5FPMeter(),
        # metrics.RocAucMeter('wAUC', ':4.3f'),
        _defs.metrics.wAUCMeter(),
    ]

    # Create writer
    pred_writer = _defs.metrics.PredictionWriter()

    print(f'evaluate_for_dataset with suffix {suffix}')

    # Evaluate model
    evaluate_model(
        loader=loader,
        model=model,
        criterion=criterion,
        device=device,
        loss_meter=loss_meter,
        target_meters=target_meters,
        score_meters=score_meters+[pred_writer],
    )

    # write predictions
    pred_writer.write(f'model/predictions{suffix}.csv')
    return {
        meter.name: meter.avg
        for meter in [loss_meter] + target_meters + score_meters
    }


def evaluate(args):
    # Set up directory name for output directory
    experiment_dir = pathlib.Path(args['model'])
    result_file = experiment_dir / 'results.csv'

    # Get model and log directories
    # Concatenate path to one subdirectory for logging
    log_dir = experiment_dir / 'log'
    model_file = experiment_dir / 'model' / 'best_model.pt.tar'

    # Summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Decide whether to run on GPU or CPU
    if torch.cuda.is_available():
        log.info('Using GPU')
        device = torch.device('cuda')
    else:
        log.info('Using CPU, this will be slow')
        device = torch.device('cpu')

    # Seed if requested
    if args['seed']:
        log.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        _defs.seed_everything(args['seed'])

    # Data loaders
    args['pre_rotate'] = args['post_rotate'] = args['post_flip'] = args['pre_flip'] = False
    te_loader, te_dataset = get_data_loader(args['te_csv'], args)
    print(f'Evaluating on {len(te_loader)} batches')

    # Input channels
    in_channels = 1 if args['grayscale'] else 3
    in_channels += 3 if args['demosaic_oracle'] else 0
    in_channels += 1 if args['lsbr_reference'] else 0
    args['drop_rate'] = 0
    print('Evaluating with drop rate', args['drop_rate'])

    # Set up model
    model = get_b0(
        # args['network'],
        pretrained=args["pretrained"],
        in_channels=in_channels,
        # channel=args['channel'],
        shape=args['shape'],
        drop_rate=args['drop_rate'],
        no_stem_stride=args['no_stem_stride'],
    ).to(device)
    summary(
        model,
        input_size=(args['batch_size'], in_channels, *args['shape'])
    )

    # Set up loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    log.info(f'=> loaded trained model {model_file} ({checkpoint["epoch"]} epochs)')

    # Evaluation
    scores = evaluate_for_dataset(
        loader=te_loader,
        model=model,
        criterion=criterion,
        device=device,
    )

    # write columns (on first write)
    if not pathlib.Path(result_file).exists():
        with open(result_file, 'a') as f:
            f.write(
                ','.join(ARGS_COLS + SCORES_COLS) + '\n'
            )

    # write results
    print(scores)
    with open(result_file, 'a') as f:
        s = [
            str(args[c])
            for c in ARGS_COLS
        ] + [
            str(scores[c])
            for c in SCORES_COLS
        ]
        f.write(','.join(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, help='Path to model', required=True)

    # Data
    parser.add_argument('--dataset', type=str, help='Path to image root', default='/scratch/martin.benes/alaska_20230303')
    parser.add_argument('--te_csv', type=str, help='Path to csv file containing the test images', default='config/split_te.csv')
    parser.add_argument('--channel', type=int, help='Channel to predict', default=None)
    # Data: Covers
    parser.add_argument('--demosaic', nargs='+', type=str, default=None, help='Demosaicking method')
    # Data: Stegos
    parser.add_argument('--stego_method', type=str, default=None, help='Selected stego method')
    parser.add_argument('--alpha', type=float, default=None, help='Selected embedding rate (alpha)')

    # Evaluation: parameters
    parser.add_argument('--batch_size', type=int, help='Batch size', default=None)
    # Evaluation: counter-fitting
    parser.add_argument('--drop_rate', type=float, help='Dropout rate', default=None)
    # Evaluation: run
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=0)
    parser.add_argument('--seed', type=int, help='Optionally seed everything for deterministic training.')

    # Parse args
    args = vars(parser.parse_args())
    # Update args
    with open(pathlib.Path(args['model']) / 'config.json') as f:
        config = json.load(f)
    config.update((k, v) for k, v in args.items() if v is not None)

    print(f'{json.dumps(config, indent=4)}')

    evaluate(config)
