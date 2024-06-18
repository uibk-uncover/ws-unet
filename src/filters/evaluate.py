"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import glob
import numpy as np
import pandas as pd
import pathlib
import scipy.signal
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
    filter: np.ndarray,
    filter_name: str,
    channels: typing.Tuple[int],
    process_image: typing.Callable,
    imread: typing.Callable = _defs.imread4_u8,
    **kw,
):
    resid = get_filter_residuals(
        fname=fname,
        filter=filter,
        imread=imread,
        process_image=process_image,
        **kw,
    )

    # mae
    mae = np.nanmean(np.abs(resid))

    # wmae
    x = imread(fname)
    rho = cl.hill._costmap.compute_cost(x[..., channels[0]])[..., None]
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


def get_coefficients(
    filter_name: str,
    flatten: bool = True,
) -> np.ndarray:
    # named filter
    if flatten:
        return NAMED_FILTERS[filter_name]
    else:
        return NAMED_FILTERS_2D[filter_name]


def get_filter_estimates(model_path: str) -> pd.DataFrame:
    return pd.concat([
        pd.read_csv(fname, dtype={'channels': str, 'inbayer': str})
        for fname in glob.glob(str(model_path / 'OLS_*.csv'))
    ])


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
    filter_names: typing.Tuple[np.ndarray] = ['AVG', 'KB'],
    channels: typing.Tuple[typing.Tuple[int]] = ((3,),),
    imread: typing.Callable = _defs.imread4_f32,  # reads image
    **kw,
) -> float:

    # iterate channels
    res = []
    for channel, filter_name in zip(channels, filter_names):

        # Image processor
        process_image = _defs.get_processor(channels=channel)

        # Coefficients
        filter = get_coefficients(filter_name=filter_name)

        #
        res_i = get_filter_residuals_cover(
            input_dir,
            filter_name=filter_name,
            filter=filter,
            channels=channel,
            process_image=process_image,
            imread=imread,
            **kw
        )
        res.append(res_i)

    return pd.concat(res)


if __name__ == '__main__':
    #
    NUM_IMAGES = None
    DATA_PATH = pathlib.Path('../data')
    OUTPUT_PATH = pathlib.Path('../results/prediction')
    filter_names = ['AVG', 'KB']
    channels = [[3], [3]]

    # run pixel prediction
    res = run(
        # data parameters
        DATA_PATH,
        filter_names=filter_names,
        channels=channels,
        #
        # take_num_images=NUM_IMAGES,
        # split='split_te.csv',
        progress_on=True,
    )

    # save result
    res.to_csv(OUTPUT_PATH / 'filters.csv', index=False)
