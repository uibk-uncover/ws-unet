
import argparse
import collections
import json
import numpy as np
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
# from training import setup_custom_logger, create_run_name, seed_everything
# from training.metrics import AverageMeter, MAEMeter, CorrMeter
# from training.metrics import WSMeter, WS255Meter
# from training.metrics import PredictionWriter
# from training.losses import wL1Loss, L1CorrLoss

log = setup_custom_logger(pathlib.Path(__file__).name)

ARGS_COLS = [
    'model',
    'dataset',
    'te_csv',
    # 'quality',
    # 'stego_method',
    'drop_rate',
]
SCORES_COLS = [
    'loss',
    'mae',
    # 'mae0',
    # 'mae1',
    # 'corr',
    'ws'
]


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
            # noises = model(images)
            # outputs = images + noises

            # Collect input dropout mask
            # dropout_mask = model.input_dropout.mask.bool()
            dropout_mask = None

            # Compute loss
            loss = criterion(outputs, targets, images)

            # Record performance
            images = images.cpu().numpy()
            covers = covers.cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            # dropout_mask = dropout_mask.cpu().numpy()
            # print(dropout_mask.sum(), model.input_dropout.p)
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
    # print(x_.size(), x_[0, 0, 1:4, 1:4]*255.)

    # infere
    y_ = model(x_)
    # print(y_[0, 0, 1:4, 1:4])
    # print('output:', y_.size(), y_[0, 0, 1:4, 1:4]*255.)

    # convert back to numpy
    y = y_.detach().numpy()[0, 0, 1:-1, 1:-1] * 255.
    # print(y[:3, :3])
    return y[..., None]


def evaluate_for_dataset(loader, model, criterion, device, suffix):

    # Create meters
    loss_meter = metrics.AverageMeter('Loss', ':.4e')
    mask_meters = [
        metrics.MAEMeter('MAE', ':.4e')
    ]
    # mask_meters = collections.OrderedDict([
    #     ('mae', metrics.MAEMeter('MAE', ':.4e')),
    #     ('mae0', metrics.MAEMeter('MAE0', ':.4e', masked=False)),
    #     ('mae1', metrics.MAEMeter('MAE1', ':.4e', masked=True)),
    #     ('corr', metrics.CorrMeter('Corr', ':.4e'))
    # ])
    target_meters = [
        # metrics.WSMeter('WS', ':.4e'),
        metrics.WSMeter('WS', ':.4e'),
    ]
    # # Create writer
    # pred_writer = PredictionWriter()

    print(f'evaluate_for_dataset with suffix {suffix}')

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

    # # write predictions
    # pred_writer.write(f'model/predictions{suffix}.csv')

    return {
        'loss': loss_meter.avg,
        **({
            v.name: v.avg
            for k, v in (mask_meters | target_meters).items()
        }),
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
        seed_everything(args['seed'])

    # Data loaders
    args['pre_rotate'] = args['post_rotate'] = args['post_flip'] = args['pre_flip'] = False
    te_loader, te_dataset = get_data_loader(args['te_csv'], args)
    print(f'Evaluating on {len(te_loader)} batches')

    # Input channels
    in_channels = 1 if args['grayscale'] else 3
    in_channels += 1 if args['parity_oracle'] else 0
    in_channels += 3 if args['demosaic_oracle'] else 0
    # in_channels += 4 if args['demosaic_oracle'] else 0
    out_channels = 1  # if args['network'] == 'unet' else 2
    # print('Evaluating with drop rate', args['drop_rate'])

    # Set up model
    model = get_model(
        args['network'],
        in_channels=in_channels,
        out_channels=out_channels,
        channel=args['channel'],
        drop_rate=0.0,
    ).to(device)
    summary(
        model,
        input_size=(args['batch_size'], in_channels, *args['shape'])
    )

    # Set up loss and optimizer
    if args['loss'] == 'l1':
        criterion = losses.L1Loss().to(device)
    elif args['loss'] == 'ws':
        criterion = losses.L1WSLoss().to(device)
    elif args['loss'] == 'l1ws':
        criterion = losses.L1WSLoss().to(device)
    else:
        raise NotImplementedError(f'loss {args["loss"]} not implemented')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    # checkpoint['state_dict']['input_dropout.p'] = torch.tensor(args['drop_rate'])
    # model.load_state_dict(checkpoint['state_dict'])
    # model.drop_rate = args['drop_rate']
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

    # Convert parameters
    config['filters_cover'] = {
        'height': config['shape'][0],
        'width': config['shape'][1],
        'quality': config['quality']
    }
    config['filters_cover_oneof'] = {
        'demosaic': config['demosaic']
    }
    config['filters_stego'] = {
        'stego_method': config['stego_method'],
        'alpha': config['alpha'],
    }

    evaluate(config)
