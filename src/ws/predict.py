"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import glob
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib
import re
import scipy.signal
import seaborn as sns
import stegolab2 as sl2
import sys
import torch
import torchvision.transforms as transforms
import typing
sys.path.append('.')
import _defs
import diffusion
import fabrika
import filters
# from kb.predict import get_coefficients, NAMED_FILTERS


NAMED_FILTERS = {
    'KB': np.array([[
        [-1, +2, -1],
        [+2,  0, +2],
        [-1, +2, -1],
    ]], dtype='float32').T / 4.,
    'AVG': np.array([[
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]], dtype='float32').T / 8.,
    'AVG9': np.array([[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]], dtype='float32').T / 9.,
    '1': np.array([[
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]], dtype='float32').T / 1.,
}

def attack(
    fname: str,
    demosaic: str,
    channels: typing.List[int],
    pixel_estimator: typing.Union[np.ndarray, typing.Callable],
    mean_estimator: np.ndarray = NAMED_FILTERS['AVG'],
    correct_bias: bool = False,
    weighted: bool = 1,
    imread: typing.Callable = None,
    process_image: typing.Callable = None,
    **kw
) -> float:
    """

    TODO: DDE removes edge elements in many places.

    Args:
        fname (str): Input image path.
        pixel_estimator (np.ndarray): Kernel to estimate pixel value.
        mean_estimator (np.ndarray): Kernel to estimate mean for variance.
        correct_bias (bool): Whether to correct for bias.
        imread (): Function to load the image.
        process_image (): Function to process the image with.
    """
    # read image
    x = imread(fname)

    # cover with LSB flipped
    x_bar = x ^ 1

    # process image
    x = process_image(x)
    x_bar = process_image(x_bar)

    # estimate pixel value from its neighbors
    x1_hat = pixel_estimator(x)

    # compute weights - TODO: use all channels
    if abs(int(weighted)) == 1:
        mu = scipy.signal.convolve(x[..., :1], mean_estimator[..., ::-1], mode='valid')
        mu2 = scipy.signal.convolve(x[..., :1]**2, mean_estimator[..., ::-1], mode='valid')
        var = mu2 - mu**2

        # weighted
        if int(weighted) == 1:
            weights = 1 / (5 + var)
        # anti-weighted
        else:
            weights = 5 + var  # 5 + var

        # normalize
        weights = weights / np.sum(weights)

    # unweighted - all 1s
    else:
        weights = np.ones_like(x1_hat) / x1_hat.size

    #
    x1 = x[1:-1, 1:-1, :1]
    x1_bar = x_bar[1:-1, 1:-1, :1]

    # estimate payload
    try:
        beta_hat = np.sum(
            weights * (x1 - x1_bar) * (x1 - x1_hat),
        )
        beta_hat = np.clip(beta_hat, 0, None)
        # print(f'beta: {beta_hat} [{alpha/2 if not np.isnan(alpha) else 0}]')
    except ValueError:
        beta_hat = None

    # compute bias
    if correct_bias:
        x_bias = pixel_estimator(x_bar - x)
        # x_bias = scipy.signal.convolve(x_bar - x, NAMED_FILTERS['KB'][..., ::-1], mode='valid')[..., :1]
        beta_hat -= beta_hat * np.sum(weights * (x1 - x1_bar) * x_bias)

    return {
        # 'alpha': alpha,
        **kw,
        'demosaic': demosaic,
        'beta_hat': beta_hat,
        'channels': ''.join(map(str, channels)),
        'weighted': weighted,
        'correct_bias': correct_bias,
    }


@fabrika.precovers(iterator='joblib', ignore_missing=True, n_jobs=4)  # os.cpu_count())
def attack_cover(*args, **kw):
    return attack(*args, **kw)


@fabrika.stego_spatial(iterator='joblib', ignore_missing=True, n_jobs=4)  # os.cpu_count())
def attack_stego(*args, **kw):
    return attack(*args, **kw)


# def get_filter_estimates(model_path: str) -> pd.DataFrame:
#     return pd.concat([
#         pd.read_csv(fname, dtype={'channels': str, 'inbayer': str})
#         for fname in glob.glob(str(model_path / 'OLS_*.csv'))
#     ])


# def get_filter_estimator(
#     model_path: str,
#     channels: typing.Tuple[int],
#     *,
#     model_name: str = 'KB',
#     df: pd.DataFrame = None,
# ) -> np.ndarray:
#     # named filter
#     if model_name in NAMED_FILTERS:
#         kernel = NAMED_FILTERS[model_name]

#     # OLS filter
#     else:
#         # load data
#         if df is None:
#             df = get_filter_estimates(model_path)
#         channels_name = ''.join(map(str, channels))
#         df_c = df[df.channels == channels_name]

#         # aggregate data
#         beta_columns = [
#             f'beta_{i}'
#             for i in _defs.BETAS_PER_MODEL['gray'][:-1]
#         ]
#         beta_hat = df_c[beta_columns].median().to_dict()

#         # construct kernel
#         rgx = r'^beta_([xyz])([0-9])([0-9])$'
#         C = len(np.unique([re.sub(rgx, r'\g<1>', k) for k in beta_hat]))
#         H = np.max([int(re.sub(rgx, r'\g<2>', k)) for k in beta_hat])
#         W = np.max([int(re.sub(rgx, r'\g<3>', k)) for k in beta_hat])
#         labels = [re.sub(rgx, r'\g<1>\g<2>\g<3>', k) for k in beta_hat]
#         kernel = np.zeros((H+1, W+1, C), dtype='float64')
#         for label in labels:
#             c = ord(label[0]) - ord('x')
#             x = int(label[1])
#             y = int(label[2])
#             kernel[x, y, c] = beta_hat[f'beta_{label}']

#     def predict(x):
#         y = scipy.signal.convolve(
#             x / 255., kernel[..., ::-1],
#             mode='valid',
#         )[..., :1]
#         return y * 255.

#     return predict


def run_ws(
    input_dir: pathlib.Path,
    stego_method: str,
    alpha: float,
    demosaic: str,
    kernel_path: pathlib.Path,
    model_name: str,
    model_path: str,
    channels: typing.Tuple[int],
    imread: typing.Callable = _defs.imread4_u8,  # reads image
    **kw,
) -> float:
    """"""
    #
    df = filters.get_filter_estimates(kernel_path)

    # iterate channels
    # res = []
    # for c, model_name in zip([(3,)]*len(model_names), model_names):
    # c = (3,)

    # channel data
    channels_name = ''.join(map(str, channels))
    # df_c = df[df.channels == channels_name]

    # image processor
    process_cover = _defs.get_processor_2d('gray', channels=channels)

    # pixel estimator
    if model_name in NAMED_FILTERS or model_name.startswith('OLS'):
        if model_name not in NAMED_FILTERS:
            model_name = f'OLS_{channels_name}'
        pixel_estimator = filters.get_filter_estimator(
            model_path=kernel_path,
            model_name=model_name,
            channels=channels,
            df=df,
        )
    else:  # model_name == 'UNet':
        pixel_estimator = diffusion.get_unet_estimator(
            model_path=model_path,
            model_name=model_name,
            # used same image loader
            channels=channels,
        )
        model_name = 'UNet'

    # attack configuration
    if stego_method:
        attack = attack_stego
        kw_attack = {
            'stego_method': stego_method,
            'alpha': alpha,
        }
    else:
        attack = attack_cover
        kw_attack = {}

    # run WS attack
    res = attack(
        input_dir,
        demosaic=demosaic,  # 'linear',
        inbayer=None,
        **kw_attack,
        pixel_estimator=pixel_estimator,
        mean_estimator=NAMED_FILTERS['AVG'],
        model_name=model_name,
        channels=channels,
        process_image=process_cover,
        imread=imread,
        **kw
    )
    res['channels'] = channels_name
    res = res[~res.beta_hat.isna()]
    return res
    # res.append(res_i)

    # return pd.concat(res)


if __name__ == '__main__':
    BOSS_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2024-01-26')
    # BOSS_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18')
    NETWORK = 'unet_2'
    STEGO_METHODS = [None, 'LSBr']  # None, 'LSBr', 'HILLr'
    ALPHAS = [.4, .2, .1]  # .4, .2, .1, .05, .01
    MODEL_NAMES = ['AVG', 'KB']  # 'AVG', 'KB', 'OLS', 'UNet'
    model = 'gray'
    take_num_images = 1000
    matched = True
    matched_s = '' if matched else '_mismatch'
    # take_num_images = None

    res = []
    for stego_method in STEGO_METHODS:
        for alpha in ALPHAS if stego_method else [.0]:
            for model_name in MODEL_NAMES:
                print(model_name, stego_method, alpha)
                res_i = run_ws(
                    input_dir=BOSS_PATH,
                    stego_method=stego_method,
                    alpha=alpha,
                    demosaic=None,
                    channels=(3,),
                    #
                    model=model,
                    model_path=None,
                    kernel_path=pathlib.Path(f'../results/filters_boss/{model}'),
                    model_name=model_name,
                    correct_bias=False,
                    weighted=0,
                    #
                    split='split_te.csv',
                    take_num_images=take_num_images,
                    shuffle_seed=12345,
                    progress_on=True,
                )
                res.append(res_i)
                # print(res_i)  # .to_string())

    # images
    for stego_method in STEGO_METHODS:
        for alpha in ALPHAS if stego_method else [.0]:
            print('UNet', stego_method, alpha, 'l1ws')

            # models
            # stego = stego_method if stego_method and loss == 'l1ws' else 'dropout'
            model_stego_method = STEGO_METHODS[1] if matched else 'LSBr'
            model_path = f'/gpfs/data/fs71999/uncover_mb/experiments/ws/{model_stego_method}'
            model_name = diffusion.get_model_name(
                network=NETWORK,
                stego_method=model_stego_method,
                alpha=.4,
                drop_rate=.0,
                loss='l1ws',
            )
            res_i = run_ws(
                input_dir=BOSS_PATH,
                stego_method=stego_method,
                alpha=alpha,
                demosaic=None,
                #
                model=model,
                model_path=model_path,
                kernel_path=pathlib.Path(f'../results/filters_boss/{model}'),
                channels=(3,),
                model_name=model_name,
                correct_bias=False,
                weighted=0,
                #
                split='split_te.csv',
                take_num_images=take_num_images,
                progress_on=True,
            )
            res_i['model_name'] = 'UNet_L1ws'
            res.append(res_i)
            # print(res_i)

            # models
            for drop_rate in [.1]:  # [.0, .1]:
                print(model_name, stego_method, alpha, 'l1')
                model_path = '/gpfs/data/fs71999/uncover_mb/experiments/ws/dropout'
                model_name = diffusion.get_model_name(
                    network=NETWORK,
                    stego_method='dropout',
                    alpha=None,
                    drop_rate=drop_rate,
                    loss='l1',
                )
                res_i = run_ws(
                    input_dir=BOSS_PATH,
                    stego_method=stego_method,
                    alpha=alpha,
                    demosaic=None,
                    #
                    model=model,
                    model_path=model_path,
                    kernel_path=pathlib.Path(f'../results/filters_boss/{model}'),
                    channels=(3,),
                    model_name=model_name,
                    correct_bias=False,
                    weighted=0,
                    #
                    split='split_te.csv',
                    take_num_images=take_num_images,
                    progress_on=True,
                    shuffle_seed=12345,
                )
                res_i['model_name'] = f'UNet_L1_{drop_rate}'
                res.append(res_i)
                # print(res_i)
    #
    res = pd.concat(res).reset_index(drop=True)
    if 'stego_method' in res:
        res['stego_method'] = res['stego_method'].fillna('Cover')
    else:
        res['stego_method'] = 'Cover'
    # res.to_csv('out.csv', index=False)
    # res = pd.read_csv('out.csv')

    # save error
    res['err'] = (res['beta_hat'] - res['alpha']/2).abs()
    res_err = (
        res.groupby(['alpha', 'model_name'])
        .agg({'err': [
            'mean',
            'median',
        ]})
        .reset_index(drop=False)
    )
    res_err.to_csv(f'../text/img/ws_err_{STEGO_METHODS[1]}{matched_s}_alaska.csv', index=False)

    #
    res['alpha'] = res['alpha'].fillna(0.)
    res['alpha'] = res['alpha'].apply(lambda a: f'{a:.2f}')

    # plot
    g = sns.FacetGrid(
        res, col="stego_method",
        sharex=False,
        gridspec_kws={'width_ratios': [1]+[len(ALPHAS)]*(len(STEGO_METHODS)-1)},
    )
    g.map_dataframe(
        sns.boxplot,
        x='alpha', y='beta_hat', hue="model_name",
        showmeans=True, showfliers=False,
        meanprops={'markerfacecolor': 'red', 'markeredgecolor': 'red'},
    )
    g.add_legend()
    for ax in g.axes[0]:
        for alpha in ALPHAS:
            ax.axhline(alpha/2, color='yellow', linewidth=.5)
    outfile = pathlib.Path(f'../text/img/ws_grayscale_boxplot_{STEGO_METHODS[1]}{matched_s}_alaska.png')
    g.savefig(outfile, dpi=600, bbox_inches='tight')
    print(f'Output saved to {outfile.absolute()}')

    res_box = (
        res.groupby(['alpha', 'model_name'])
        .agg({'beta_hat': [
            'min',
            _defs.iqr_interval(.25, sign=-1.5),
            _defs.quantile(.25),
            _defs.quantile(.5),
            _defs.quantile(.75),
            _defs.iqr_interval(.75, sign=1.5),
            'max',
        ]})
    )
    res_box.columns = [col[1] for col in res_box.columns.values]
    res_box = res_box.reset_index().sort_values(['alpha', 'model_name'])
    res_box.to_csv(f'../text/img/predict_ws_{STEGO_METHODS[1]}{matched_s}_alaska_box.csv', index=False)
    print(res_box)
    # print(res)
    # res['stem'] = res['name'].apply(lambda f: pathlib.Path(f).stem)
    # res = res.pivot(index='stem', columns=('stego_method', 'alpha', 'model_name'), values=('beta_hat'))
    # res = res.reset_index(drop=True)
    # res.columns = ['_'.join(map(str, col)).strip() for col in res.columns.values]
    # res.to_csv(f'../text/img/predict_ws_{STEGO_METHODS[1]}.csv', index=False)
    # print(res)
