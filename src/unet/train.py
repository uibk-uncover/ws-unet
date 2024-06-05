
import argparse
import json
import numpy as np
import os
import pathlib
import shutil
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
# from torchscan import summary

sys.path.append('.')
from data import get_data_loader
from evaluate import evaluate_model
from model import get_model
sys.path.append('..')
from _defs import seed_everything, setup_custom_logger, create_run_name
from _defs import metrics, losses


log = setup_custom_logger(pathlib.Path(__file__).name)

NUM_ACCUMULATION_STEPS = 1


def train_one_epoch(
    tr_loader,
    model,
    criterion,
    optimizer,
    epoch,
    writer,
    device,
    args
):
    """"""
    # Create meters
    loss_meter = metrics.LossMeter(':.4e')
    mask_meters = [
        metrics.MAEMeter(),
        # metrics.MAEMeter('MAE0', ':.4e', masked=False),
        # metrics.MAEMeter('MAE1', ':.4e', masked=True),
        # metrics.CorrMeter(),
    ]
    target_meters = [
        # metrics.WSMeter('WS', ':.4e'),
        metrics.WSMeter(),
    ]
    progress = metrics.ProgressMeter(
        len(tr_loader),
        [loss_meter, *mask_meters, *target_meters],
        prefix='Epoch: [{}]'.format(epoch),
    )

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    # print(model.e11.weight.data)  #, model.e11.bias.data)

    for i, (images, targets) in enumerate(tr_loader):
        batch_size = images.size(0)

        # Move data to the same device as model
        images = images.to(device, non_blocking=True)
        targets[0] = targets[0].to(device, non_blocking=True)
        alphas = targets[1].numpy()
        targets[1] = targets[1].to(device, non_blocking=True)
        covers = targets[0]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward-propagate
        # if i > 0 and i % args['print_freq'] == 0 or i == len(tr_loader)-1:
        #     print(images[0, 0, :5, :5])
        outputs = model(images)
        # if i > 0 and i % args['print_freq'] == 0 or i == len(tr_loader)-1:
        #     print(outputs[0, 0, :5, :5])
        # noises = model(images)
        # outputs = images + noises

        # Collect input dropout mask
        # dropout_mask = model.input_dropout.mask.bool()
        dropout_mask = None

        # Compute loss
        # print(model.e11.weight, model.e11.bias)
        loss = criterion(outputs, targets, images)

        # Record performance
        # images = images[:, :, 1:-1, 1:-1]
        # covers = covers[:, :, 1:-1, 1:-1]
        images = images.cpu().numpy()
        covers = covers.cpu().numpy()
        logits = outputs.detach().cpu().numpy()
        # dropout_mask = dropout_mask.cpu().numpy()
        loss_meter.update(loss.item(), batch_size)
        for meter in mask_meters:
            meter.update(
                covers,
                logits,
                dropout_mask,
            )
        for meter in target_meters:
            meter.update(
                images,
                logits,
                alphas,
            )

        # Compute gradients
        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()

        # # Manually disable center pixels in L1.1
        # model.disable_center_pixels()

        # print(i, '/', len(tr_loader))
        # print(model.e11.weight.grad)

        # Gradient descend
        optimizer.step()
        # if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(tr_loader)):
        #     optimizer.step()
        #     optimizer.zero_grad()

        if i > 0 and i % args['print_freq'] == 0 or i == len(tr_loader)-1:
            log.info(progress.to_str(batch=i))
            # criterion(outputs, targets, dropout_mask, debug=True)
    # print(model.e11.weight.data)#, model.e11.bias.data)

    for meter in [loss_meter] + mask_meters + target_meters:
        writer.add_scalar('train/' + meter.name, meter.avg, global_step=epoch)


def validate(va_loader, model, criterion, writer, device, epoch=None):
    """"""
    # Create meters
    loss_meter = metrics.LossMeter(':.4e')
    mask_meters = [
        metrics.MAEMeter(),
        # metrics.MAEMeter('MAE0', ':.4e', masked=False),
        # metrics.MAEMeter('MAE1', ':.4e', masked=True),
        # metrics.CorrMeter('Corr', ':.4e'),
    ]
    target_meters = [
        metrics.WSMeter(),
        # metrics.WS255Meter('WS255', ':.4e'),
    ]
    progress = metrics.ProgressMeter(
        len(va_loader),
        [loss_meter, *mask_meters, *target_meters],
        prefix='Val: ',
    )

    # Evaluate model
    evaluate_model(
        loader=va_loader,
        model=model,
        criterion=criterion,
        device=device,
        loss_meter=loss_meter,
        mask_meters=mask_meters,
        target_meters=target_meters,
    )

    # Log progress
    log.info(progress.to_str(batch=0))

    # Write score
    for meter in [loss_meter] + mask_meters + target_meters:
        writer.add_scalar('val/' + meter.name, meter.avg, global_step=epoch)

    return loss_meter.avg


def train(args):
    # Set up directory name for output directory
    # experiment_dir_name = time.strftime('%Y_%m_%d_%H_%M_%S') + '-prototyping'
    experiment_dir_name = (
        time.strftime('%y%m%d%H%M%S') + '-' +
        args['SLURM_JOB_ID'] + '-' +
        create_run_name(args)
    )
    if args['experiment_dir_suffix']:
        experiment_dir_name = experiment_dir_name + '_' + args['experiment_dir_suffix']
    print(f'{experiment_dir_name=}')
    print(f'{NUM_ACCUMULATION_STEPS=}')

    # Create output directory for this experiment
    experiment_dir = pathlib.Path(args['output_dir']) / args['stego_method'] / experiment_dir_name
    if not experiment_dir.exists():
        experiment_dir.mkdir(exist_ok=False, parents=True)

    # Create resume directory
    if args['resume_dir']:
        resume_dir = pathlib.Path(args['resume_dir'])
    else:
        resume_dir = pathlib.Path(args['output_dir'])
    resume_dir = resume_dir / args['stego_method']

    # Dump args to file
    args_file = experiment_dir / 'config.json'
    with open(args_file, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=True)

    # Set up model and log directories
    # Concatenate path to one subdirectory for logging
    log_dir = experiment_dir / 'log'
    model_dir = experiment_dir / 'model'
    best_model_file = model_dir / 'best_model.pt.tar'
    latest_model_file = model_dir / 'latest_model.pt.tar'
    # Create subdirectories if they don't exist yet
    log_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(exist_ok=False)

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
    args_val = args.copy()
    args_val['pre_rotate'] = args_val['post_rotate'] = args_val['post_flip'] = args_val['pre_flip'] = False
    tr_loader, tr_dataset = get_data_loader(args['tr_csv'], args)
    va_loader, va_dataset = get_data_loader(args['va_csv'], args_val)

    # Input channels
    in_channels = 1 if args['grayscale'] else 3
    in_channels += 1 if args['parity_oracle'] else 0
    in_channels += 3 if args['demosaic_oracle'] else 0
    # in_channels += 4 if args['demosaic_oracle'] else 0
    out_channels = len(args['channel'])  # if args['network'] == 'unet' else 2

    # Set up model
    model = get_model(
        args['network'],
        in_channels=in_channels,
        out_channels=out_channels,
        channel=args['channel'],
        drop_rate=args['drop_rate'],
    ).to(device)
    summary(
        model,
        (args["batch_size"], in_channels, *args["shape"]),
    )

    # Set up loss and optimizer
    if args['loss'] == 'l1':
        criterion = losses.L1Loss().to(device)
    elif args['loss'] == 'l2':
        criterion = losses.L2Loss().to(device)
    elif args['loss'] == 'ws':
        criterion = losses.WSLoss().to(device)
    elif args['loss'] == 'l1ws':
        criterion = losses.L1WSLoss(lmbda=args['loss_lambda']).to(device)
    elif args['loss'] == 'l2ws':
        criterion = losses.L2WSLoss(lmbda=args['loss_lambda']).to(device)
    elif args['loss'] == 'l1corrws':
        criterion = losses.L1CorrLoss().to(device)
    else:
        raise NotImplementedError(f'loss {args["loss"]} not implemented')
    optimizer = torch.optim.AdamW(model.parameters(), args["learning_rate"])
    scheduler = None

    start_epoch = 0
    best_val_loss = np.inf
    patience = args['patience']

    if args['resume']:
        resume_model_file = resume_dir / args['resume'] / 'model' / 'best_model.pt.tar'
        print(resume_model_file)
        if resume_model_file.exists():
            log.info("=> loading checkpoint '{}'".format(args["resume"]))

            checkpoint = torch.load(resume_model_file, map_location=device)

            # start_epoch = checkpoint['epoch']
            # best_val_loss = checkpoint['best_val_loss']

            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args["resume"]))
        else:
            raise Exception("no checkpoint found at '{}'".format(args["resume"]))

    # Training loop
    for epoch in range(start_epoch, args["num_epochs"]):
        # Reshuffle training dataset
        tr_dataset.reshuffle()

        # Train for one epoch
        train_one_epoch(
            tr_loader=tr_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            device=device,
            args=args
        )

        val_loss = validate(
            va_loader=va_loader,
            model=model,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch
        )

        if scheduler:
            scheduler.step(val_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'patience': patience,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }, latest_model_file)

        # Remember best validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            patience = args['patience']
            shutil.copyfile(latest_model_file, best_model_file)
            print('best model!', val_loss, 'is better than', best_val_loss)
            best_val_loss = val_loss
        else:
            patience -= 1
            print('patience countdown:', patience)

        # Early stopping
        if patience <= 0:
            print('my patience is over, early stopping!')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--dataset', type=str, help='Path to image root', default='/scratch/martin.benes/alaska_20230303')
    parser.add_argument('--tr_csv', type=str, help='Path to csv file containing the training images', default='split_tr.csv')
    parser.add_argument('--va_csv', type=str, help='Path to csv file containing the validation images', default='split_va.csv')
    parser.add_argument('--channel', nargs='+', type=int, default=[0], help='Channel to predict')
    # Data: Covers
    parser.add_argument('--shape', nargs='+', type=int, default=[512, 512], help='Dataset shape')
    parser.add_argument('--demosaic', nargs='+', type=str, default=None, help='Demosaicking method')
    parser.add_argument('--quality', type=int, default=None, help='Selected specific JPEG quality')
    # Data: Stegos
    parser.add_argument('--covers_only', action='store_true', help='Whether to train on covers only')
    parser.add_argument('--stego_method', type=str, default=None, help='Selected stego method')
    parser.add_argument('--alpha', type=float, default=None, help='Selected embedding rate (alpha)')
    parser.add_argument('--stego_methods', nargs='+', type=str, default=None, help='Selected stego method')
    parser.add_argument('--alphas', nargs='+', type=float, default=None, help='Selected embedding rate (alpha)')
    parser.add_argument('--rotation', type=int, help='Dataset rotation to use')

    # Training: model
    parser.add_argument('--network', type=str, help='Network type', default='unet_1')
    parser.add_argument('--grayscale', action='store_true', help='Train only on luminance')
    parser.add_argument('--parity_oracle', action='store_true', help='Whether model knows the parity')
    parser.add_argument('--demosaic_oracle', action='store_true', help='Whether model knows the demosaic method')
    parser.add_argument('--resume_dir', type=str, help='Path to checkpoints')
    parser.add_argument('--resume', type=str, help='Model from which to resume training')
    # Training: parameters
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=150)
    parser.add_argument('--pair_constraint', action='store_true', help='Use pair-constraint training')
    # Training: counter-overfitting
    parser.add_argument('--drop_rate', type=float, help='Dropout rate', default=0.0)
    parser.add_argument('--post_flip', action='store_true', help='Augment with random hv flipping (post embedding)')
    parser.add_argument('--pre_flip', action='store_true', help='Augment with random rotation (pre embedding)')
    parser.add_argument('--post_rotate', action='store_true', help='Augment with random rotation (post embedding)')
    parser.add_argument('--pre_rotate', action='store_true', help='Augment with random rotation (pre embedding)')
    # Training: run
    parser.add_argument('--loss', type=str, default='l1ws', help='loss function')
    parser.add_argument('--loss_lambda', type=float, default=.25, help='loss weight')
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=8)
    parser.add_argument('--seed', type=int, help='Optionally seed everything for deterministic training.')
    parser.add_argument('--patience', type=int, help="Stop training if validation loss has not improved for X epochs", default=5)
    parser.add_argument('--debug', action='store_true', help='If test run, ignored in model search')

    # Output
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--experiment_dir_suffix', type=str, help='Suffix for output directory')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')

    # Parse args
    args = vars(parser.parse_args())

    # Convert parameters
    args['filters_cover'] = {
        'height': args['shape'][0],
        'width': args['shape'][1],
        'quality': args['quality']
    }
    args['filters_cover_oneof'] = {
        'demosaic': args['demosaic']
    }
    args['filters_stego'] = {
        'stego_method': args['stego_method'],
        'alpha': args['alpha'],
    }
    args['filters_stego_oneof'] = {
        'stego_method': args['stego_methods'],
        'alpha': args['alphas'],
    }
    if args['covers_only']:
        args['stego_method'] = 'dropout'
        args['stego_methods'] = None
        args['alpha'] = None
        args['alphas'] = None
    else:
        if args['stego_method'] is None and args['stego_methods'] is not None:
            args['stego_method'] = '_'.join(args['stego_methods'])
            args['stego_methods'] = None
        if args['alpha'] is None and args['alphas'] is not None:
            args['alpha'] = '_'.join([f'{alpha:.3f}' for alpha in args['alphas']])
            args['alphas'] = None

    # Add SLURM job id
    args['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID', 'noslurm')

    print(f'{json.dumps(args, indent=4)}')
    train(args)
