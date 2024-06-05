
import joblib
import logging
import numpy as np
import os
import random
import sys
import torch
import tqdm
import typing


class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, total: int = None, **kwargs):
        with tqdm.tqdm(total=total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        # self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def setup_custom_logger(name):
    # Taken from https://stackoverflow.com/questions/7621897/python-logging-module-globally
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_run_name(args: typing.Dict[str, typing.Any]) -> str:
    run_name = str(args['network'])
    if 'no_stem_stride' in args and args['no_stem_stride']:
        run_name += '-nostride'
    run_name += '-'
    if args['alpha']:
        run_name += 'alpha_'
        run_name += str(args['alpha']) + '_'
    if args['grayscale']:
        run_name += 'grayscale_'
    else:
        run_name += 'color'
        run_name += '_' + ''.join(map(str, args['channel']))
    if args['demosaic']:
        run_name += '_'.join(args['demosaic']) + '_'
    if args['demosaic_oracle']:
        run_name += 'oracle_'
    if args['loss']:
        run_name += args['loss'] + '_'
        if args['loss'] == 'l1ws':
            run_name += f'{args["loss_lambda"]:.02f}_'
    if args['learning_rate']:
        run_name += 'lr_'
        run_name += str(args['learning_rate']) + '_'
    if args['drop_rate']:
        run_name += 'dr_'
        run_name += str(args['drop_rate'])
    return run_name


def quantile(n):
    def q_(x):
        return x.quantile(n)
    q_.__name__ = f'q_{n*100:.0f}'
    return q_


def iqr_interval(n, sign=1):
    def iqr(x):
        return x.quantile(.75) - x.quantile(.25)

    def iqr_interval_(x):
        return (x.quantile(n) + sign * iqr(x)).clip(x.min(), x.max())

    iqr_interval_.__name__ = f'q_{n*100:.0f}_iqr'
    return iqr_interval_
