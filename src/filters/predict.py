"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import glob
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import re
import scipy.signal
import stegolab2 as sl2
import sys
import typing

sys.path.append('.')
import fabrika
import _defs


NAMED_FILTERS = {
    'KB': np.array([
        [-1], [+2], [-1], [+2], [-1], [+2], [-1], [+2],
    ], dtype='float64') / 4.,
    'AVG': np.ones((8, 1)) / 8.,
}

NAMED_FILTERS_2D = {
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


def get_filter_residuals(
    fname: str,
    filter: np.ndarray,
    process_image: typing.Callable,
    imread: typing.Callable = _defs.imread4_u8,  # reads image
    **kw,
) -> typing.Tuple:

    # read image
    img = imread(fname)

    # process images
    img = process_image(img)

    #
    x, y = img[..., :-1], img[..., -1:]

    # compute prediction
    y_hat = x @ filter

    # compute residual
    resid = y - y_hat

    return resid


@fabrika.precovers(iterator='joblib', convert_to='pandas', ignore_missing=True, n_jobs=5)  # n_jobs=os.cpu_count())
def get_filter_residuals_cover(
    fname: str,
    filter_name: str,
    channels: typing.Tuple[int],
    process_image: typing.Callable,
    imread: typing.Callable = _defs.imread4_u8,
    **kw,
):
    resid = get_filter_residuals(
        fname=fname,
        imread=imread,
        process_image=process_image,
        **kw,
    )

    # mae
    mae = np.nanmean(np.abs(resid))

    # wmae
    x = imread(fname)
    rho = sl2.hill.compute_rho(x[..., channels[0]])[..., None]
    rho[np.isinf(rho) | np.isnan(rho) | (rho > 10**10)] = 10**10
    rho = process_image(np.repeat(rho, 4, axis=2))[..., -1:]
    wmae = np.nanmean(np.abs(resid)[rho <= np.quantile(rho, .1)])
    # print(mae, wmae)

    # result
    channels_name = ''.join(map(str, channels))
    return {
        'fname': fname,
        f'mae_{channels_name}_{filter_name}': mae,
        f'wmae_{channels_name}_{filter_name}': wmae,
        **kw,
    }


@fabrika.stego_spatial(iterator='joblib', convert_to='pandas', ignore_missing=True, n_jobs=5)  # n_jobs=os.cpu_count())
def get_filter_residuals_stego(
    fname: str,
    filter_name: str,
    channels: typing.Tuple[int],
    **kw,
):
    resid = get_filter_residuals(fname=fname, **kw)
    mae = np.nanmean(np.abs(resid))

    # result
    channels_name = ''.join(map(str, channels))
    return {
        'fname': fname,
        f'mae_{channels_name}_{filter_name}': mae,
        # f'wae_{channels_name}_{kernel_name}': wmae,
        **kw,
    }


def get_coefficients(
    # model: str,
    model_name: str,
    flatten: True,
    # df: pd.DataFrame = None,
) -> np.ndarray:
    # named filter
    # if kernel_name in NAMED_FILTERS:
    if flatten:
        return NAMED_FILTERS[model_name]
    else:
        return NAMED_FILTERS_2D[model_name]

    # beta_columns = [
    #     f'beta_{i}'
    #     for i in _defs.BETAS_PER_MODEL[model][:-1]
    # ]

    # beta_hat = df[beta_columns].mean()
    # beta_hat = beta_hat.to_numpy()[..., None]

    # return beta_hat


def get_filter_estimates(model_path: str) -> pd.DataFrame:
    return pd.concat([
        pd.read_csv(fname, dtype={'channels': str, 'inbayer': str})
        for fname in glob.glob(str(model_path / 'OLS_*.csv'))
    ])


# def get_filter(
#     # model_path: str,
#     channels: typing.Tuple[int],
#     *,
#     model_name: str = 'KB',
#     df: pd.DataFrame = None,
# ) -> np.ndarray:
#     # named filter
#     if model_name in NAMED_FILTERS_2D:
#         kernel = NAMED_FILTERS_2D[model_name]

#     # # OLS filter
#     # else:
#     #     # load data
#     #     if df is None:
#     #         df = get_filter_estimates(model_path)
#     #     channels_name = ''.join(map(str, channels))
#     #     # print(df)
#     #     df_c = df[df.channels == channels_name]
#     #     # print(channels_name)
#     #     # print(df_c)

#     #     # aggregate data
#     #     beta_columns = [
#     #         f'beta_{i}'
#     #         for i in _defs.BETAS_PER_MODEL['gray'][:-1]
#     #     ]
#     #     beta_estimates = df_c[beta_columns]
#     #     beta_hat = beta_estimates.mean().to_dict()
#     #     # print(beta_estimates.mean())

#     #     # construct kernel
#     #     rgx = r'^beta_([xyz])([0-9])([0-9])$'
#     #     C = len(np.unique([re.sub(rgx, r'\g<1>', k) for k in beta_hat]))
#     #     H = np.max([int(re.sub(rgx, r'\g<2>', k)) for k in beta_hat])
#     #     W = np.max([int(re.sub(rgx, r'\g<3>', k)) for k in beta_hat])
#     #     labels = [re.sub(rgx, r'\g<1>\g<2>\g<3>', k) for k in beta_hat]
#     #     kernel = np.zeros((H+1, W+1, C), dtype='float64')
#     #     for label in labels:
#     #         c = ord(label[0]) - ord('x')
#     #         x = int(label[1])
#     #         y = int(label[2])
#     #         kernel[x, y, c] = beta_hat[f'beta_{label}']
#     #     # print(kernel.T)

#     #
#     return kernel


def infere_single(x, model):
    y = scipy.signal.convolve(
        x / 255., model[..., ::-1],
        mode='valid',
    )[..., :1]
    return y * 255.


def get_filter_estimator(*args, **kw) -> np.ndarray:
    kernel = get_coefficients(*args, flatten=True, **kw)
    return lambda x: infere_single(x, kernel)


def run(
    input_dir: pathlib.Path,
    demosaic: str,
    inbayer: str,
    model: str,
    kernel_path: pathlib.Path,
    kernel_names: typing.Tuple[np.ndarray],
    channels: typing.Tuple[typing.Tuple[int]],
    has_oracle: bool,
    imread: typing.Callable = _defs.imread4_f32,  # reads image
    **kw,
) -> float:

    df = pd.concat([
        pd.read_csv(fname, dtype={'channels': str, 'inbayer': str})
        for fname in glob.glob(str(kernel_path / 'OLS_*.csv'))
    ])

    # iterate channels
    res = []
    for c, k in zip(channels, kernel_names):

        # channel data
        channels_name = ''.join(map(str, c))
        df_c = df[df.channels == channels_name]

        # select bayer
        if inbayer is not None:
            df_c = df_c[df_c.inbayer == inbayer]
        if has_oracle:
            df_c = df_c[df_c.demosaic == demosaic]

        # Image processor
        process_image = _defs.get_processor(
            model,
            channels=c,
            inbayer=inbayer,
        )

        # kernel name
        kernel_name = k
        if kernel_name in NAMED_FILTERS:
            pass
        elif inbayer is not None:
            kernel_name = f'OLS_{channels_name}_{inbayer}'
        else:
            kernel_name = f'OLS_{channels_name}'
        if has_oracle:
            kernel_name += f'_{demosaic}'

        # get coefficients
        kernel = get_coefficients(
            model=model,
            kernel_name=k,
            df=df_c)

        # res_i = run_single_cover(
        #     input_dir,
        #     demosaic=demosaic,
        #     kernel_name=kernel_name,
        #     kernel=kernel,
        #     channels=c,
        #     inbayer=inbayer,
        #     process_image=process_image,
        #     imread=imread,
        #     **kw
        # )
        res_i = get_filter_residuals_cover(
            input_dir,
            demosaic=demosaic,
            kernel_name=kernel_name,
            kernel=kernel,
            channels=c,
            inbayer=inbayer,
            process_image=process_image,
            imread=imread,
            **kw
        )
        # res_i = get_filter_residuals_stego(
        #     input_dir,
        #     demosaic=demosaic,
        #     stego_method='LSBr',
        #     alpha=.2,
        #     kernel_name=kernel_name,
        #     kernel=kernel,
        #     channels=c,
        #     inbayer=inbayer,
        #     process_image=process_image,
        #     imread=imread,
        #     **kw
        # )
        # res_i = pd.DataFrame([row for row, resid in res_i])
        res.append(res_i)

    return pd.concat(res)


if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser(description='Run grayscale WS experiment')
    # input and output
    parser.add_argument('--input_dir', required=True, type=pathlib.Path, help='path to original dataset')
    parser.add_argument('--output_dir', default=pathlib.Path.cwd(), type=pathlib.Path, help='output_path')
    parser.add_argument('--split', default=None, type=str, help='split to use')
    parser.add_argument('--demosaic', default=None, type=str, help='demosaicking algorithm')
    # parser.add_argument('--stego_method', default=None, type=str, help='stego method')
    # parser.add_argument('--alpha', default=None, type=float, help='embedding rate')
    parser.add_argument('--has_oracle', action='store_true', help='If true, take the most adapted kernel available')
    parser.add_argument('--take_num_images', default=None, type=int, help='Take given number of images')
    parser.add_argument('--skip_num_images', default=None, type=int, help='Skip given number of images')
    # predict parameters
    parser.add_argument('--model', default='gray', type=str, help='model')
    parser.add_argument('--filter', default=None, type=str, help='filter')
    # parse args
    args = parser.parse_args()

    # check paths
    assert args.input_dir.exists(), f'{args.input_dir} does not exist'
    args.output_dir = args.output_dir / 'filters_boss' / args.model
    args.output_dir.mkdir(parents=False, exist_ok=True)

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.output_dir / "fabrika.log"),  # log to log file
            # logging.StreamHandler(),  # print to stderr
        ]
    )

    if args.model in {'gray', 'grayD'}:
        channels = ((3,),)  # (1,), (2,))
        kernel_names = tuple([args.filter]*1)  # 3)
    elif args.model in {'color1', 'color3', 'color4', 'color4D'}:
        channels = ((0, 1),)  # (1,), (2, 1))
        filter_G = args.filter if args.filter in NAMED_FILTERS else 'KB'
        kernel_names = (args.filter,)  # filter_G, args.filter)
    elif args.model in {'color8'}:
        channels = ((0, 1, 2),)
        filter_G = args.filter if args.filter in NAMED_FILTERS else 'KB'
        kernel_names = (args.filter,)  # filter_G, args.filter)
    else:
        raise NotImplementedError(f'unknown model {args.model}')
    print(f'{kernel_names=} {channels=}')

    # run pixel prediction
    # args.take_num_images = 100
    res = run(
        # data parameters
        args.input_dir,
        demosaic=args.demosaic,
        inbayer=None,
        model=args.model,
        has_oracle=args.has_oracle,
        take_num_images=args.take_num_images,
        skip_num_images=args.skip_num_images,
        # predict parameters
        kernel_path=args.output_dir,
        kernel_names=kernel_names,
        channels=channels,
        split=args.split,
        progress_on=True,
    )

    # save result
    config = f'{args.demosaic}'
    res.to_csv(args.output_dir / f'predict_{config}.csv', index=False)

    mae_cols = [c for c in res.columns if re.match(r'^mae_.*', c)]
    mae = res[mae_cols].mean()
    print(mae.to_dict())
    wmae_cols = [c for c in res.columns if re.match(r'^wmae_.*', c)]
    wmae = res[wmae_cols].mean()
    # print(wmae.to_dict())
    # print(mae)


    # print(res.columns)
    # print(mae_cols)
    # res = pd.read_csv(args.output_dir / f'ws_{config}.csv')
    # mae = {
    #     ch: res[f'mae_{ch}'].mean()
    #     for ch in ['R', 'G', 'B']
    # }
    # mse = {
    #     ch: res[f'mse_{ch}'].mean()
    #     for ch in ['R', 'G', 'B']
    # }
    # logging.info(f'prediction run with {len(res)} files: {mae.to_dict()}')
