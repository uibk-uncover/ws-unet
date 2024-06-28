"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

# import conseal as cl
# import glob
import numpy as np
# import logging
# import matplotlib.pyplot as plt
# import os
import pandas as pd
import pathlib
# import re
import scipy.signal
import seaborn as sns
import sys
# import torch
# import torchvision.transforms as transforms
import typing
sys.path.append('.')
import _defs
# import diffusion
import fabrika
import filters
import unet
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
    # demosaic: str,
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
    except ValueError:
        beta_hat = None

    # compute bias
    if correct_bias:
        x_bias = pixel_estimator(x_bar - x)
        beta_hat -= beta_hat * np.sum(weights * (x1 - x1_bar) * x_bias)

    return kw | {
        # 'demosaic': demosaic,
        'beta_hat': beta_hat,
        'channels': ''.join(map(str, channels)),
        'weighted': weighted,
        'correct_bias': correct_bias,
    }


@fabrika.precovers(iterator='joblib', ignore_missing=True, n_jobs=4)
def attack_cover(*args, **kw):
    return attack(*args, **kw)


@fabrika.stego_spatial(iterator='joblib', ignore_missing=True, n_jobs=4)
def attack_stego(*args, **kw):
    return attack(*args, **kw)


def run(
    input_dir: pathlib.Path,
    stego_method: str,
    alpha: float,
    model_name: str,
    model_path: str,
    channels: typing.Tuple[int],
    imread: typing.Callable = _defs.imread4_u8,  # reads image
    **kw,
) -> float:
    """"""

    # image processor
    process_cover = _defs.get_processor_2d(channels=channels)

    # pixel estimator
    if model_name in NAMED_FILTERS:
        pixel_estimator = filters.get_filter_estimator(
            filter_name=model_name,
            flatten=False,
        )
    else:
        pixel_estimator = unet.get_unet_estimator(
            model_path=model_path,
            model_name=model_name,
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
        # demosaic=demosaic,  # 'linear',
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
    res['channels'] = ''.join(map(str, channels))
    res = res[~res.beta_hat.isna()]
    return res


if __name__ == '__main__':
    DATA_PATH = '../data/'
    NETWORK = 'unet_2'
    L1WS_TRAIN_METHOD = 'LSBR'
    STEGO_METHODS = [None, 'LSBR']
    ALPHAS = [.4, .2, .1]
    MODEL_NAMES = ['AVG', 'KB']
    # suffix
    suffix = f'_{L1WS_TRAIN_METHOD}'

    res = []
    for stego_method in STEGO_METHODS:
        for alpha in ALPHAS if stego_method else [.0]:
            for model_name in MODEL_NAMES:
                print(model_name, stego_method, alpha)
                res_i = run(
                    input_dir=DATA_PATH,
                    stego_method=stego_method,
                    alpha=alpha,
                    demosaic=None,
                    channels=(3,),
                    #
                    model_path=None,
                    model_name=model_name,
                    correct_bias=False,
                    weighted=0,
                    #
                    progress_on=True,
                )
                res.append(res_i)

    # images
    for loss in ['l1', 'l1ws']:
        for stego_method in STEGO_METHODS:
            for alpha in ALPHAS if stego_method else [.0]:
                print('UNet', stego_method, alpha, loss)

                # models
                train_method = L1WS_TRAIN_METHOD if loss == 'l1ws' else 'dropout'
                model_path = f'../models/unet/{train_method}'
                model_name = unet.get_model_name(
                    stego_method=train_method,
                )
                res_i = run(
                    input_dir=DATA_PATH,
                    stego_method=stego_method,
                    alpha=alpha,
                    demosaic=None,
                    #
                    model_path=model_path,
                    channels=(3,),
                    model_name=model_name,
                    correct_bias=False,
                    weighted=0,
                    #
                    progress_on=True,
                )

                res_i['model_name'] = f'UNet_{loss}'
                if loss == 'l1ws':
                    res_i['model_name'] += f'_{train_method}'

                res.append(res_i)

    #
    res = pd.concat(res).reset_index(drop=True)
    if 'stego_method' in res:
        res['stego_method'] = res['stego_method'].fillna('Cover')
    else:
        res['stego_method'] = 'Cover'

    # save
    # res['err'] = (res['beta_hat'] - res['alpha']/2).abs()
    res.to_csv(f'../results/estimation/ws{suffix}.csv', index=False)

    # exit()
    # res_err = (
    #     res.groupby(['alpha', 'model_name'])
    #     .agg({'err': ['mean', 'median']})
    #     .reset_index(drop=False)
    # )
    # res_err.to_csv(f'../results/estimation/ws_err{suffix}.csv', index=False)

    # #
    # res['alpha'] = res['alpha'].fillna(0.)
    # res['alpha'] = res['alpha'].apply(lambda a: f'{a:.2f}')

    # # plot
    # g = sns.FacetGrid(
    #     res, col="stego_method",
    #     sharex=False,
    #     gridspec_kws={'width_ratios': [1]+[len(ALPHAS)]*(len(STEGO_METHODS)-1)},
    # )
    # g.map_dataframe(
    #     sns.boxplot,
    #     x='alpha', y='beta_hat', hue="model_name",
    #     showmeans=True, showfliers=False,
    #     meanprops={'markerfacecolor': 'red', 'markeredgecolor': 'red'},
    # )
    # g.add_legend()
    # for ax in g.axes[0]:
    #     for alpha in ALPHAS:
    #         ax.axhline(alpha/2, color='yellow', linewidth=.5)
    # outfile = pathlib.Path(f'../results/estimation/ws{suffix}.png')
    # g.savefig(outfile, dpi=600, bbox_inches='tight')
    # print(f'Output saved to {outfile.absolute()}')

    # res_box = (
    #     res.groupby(['alpha', 'model_name'])
    #     .agg({'beta_hat': [
    #         'min',
    #         _defs.iqr_interval(.25, sign=-1.5),
    #         _defs.quantile(.25),
    #         _defs.quantile(.5),
    #         _defs.quantile(.75),
    #         _defs.iqr_interval(.75, sign=1.5),
    #         'max',
    #     ]})
    # )
    # res_box.columns = [col[1] for col in res_box.columns.values]
    # res_box = res_box.reset_index().sort_values(['alpha', 'model_name'])
    # res_box.to_csv(f'../results/estimation/ws{suffix}.csv', index=False)
    # print(res_box)
