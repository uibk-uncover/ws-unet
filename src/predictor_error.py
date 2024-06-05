
import logging
import numpy as np
import pandas as pd
import pathlib
import stegolab2 as sl2
import typing

import _defs
import fabrika
import diffusion
import filters

@fabrika.precovers(iterator='joblib', convert_to='pandas', ignore_missing=True, n_jobs=50)  # os.cpu_count())
def attack(
    fname: str,
    demosaic: str,
    channels: typing.List[int],
    pixel_estimator: typing.Callable,
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

    # process image
    x = process_image(x)

    # estimate pixel value from its neighbors
    x1_hat = pixel_estimator(x)

    #
    x1 = x[1:-1, 1:-1, :1]

    try:
        # mae
        resid = x1 - x1_hat
        mae = np.nanmean(np.abs(x1 - x1_hat))
        # wmae
        x = imread(fname)
        rho = sl2.hill.compute_rho(x[..., channels[0]])[1:-1, 1:-1, None]
        rho[np.isinf(rho) | np.isnan(rho) | (rho > 10**10)] = 10**10
        wmae = np.nanmean(np.abs(resid)[rho <= np.quantile(rho, .1)])
    except ValueError:
        mae = None
        wmae = None

    #
    return {
        'demosaic': demosaic,
        'filter': 'UNet',
        'mae': mae,
        'wmae': wmae,
        'channels': ''.join(map(str, channels)),
        'model': 'gray',
        'inbayer': '',
        'information': 'Unconditional'
    }


def evaluate_unet(
    input_dir: pathlib.Path,
    imread: typing.Callable = _defs.imread4_u8,  # reads image
    **kw,
) -> float:
    """"""
    # iterate channels
    channels = (3,)

    stego_methods = ['dropout', 'LSBr']
    alphas = [None, .4]
    losses = ['l1', 'l1ws']
    drop_rates = [.1, .0]

    # model_paths = [
    #     '/gpfs/data/fs71999/uncover_mb/experiments/ws/dropout',
    #     '/gpfs/data/fs71999/uncover_mb/experiments/ws/LSBr'
    # ]

    # image processor
    process_cover = _defs.get_processor_2d('gray', channels=channels)

    res = []
    for stego_method, alpha, loss, drop_rate in zip(stego_methods, alphas, losses, drop_rates):
        model_path = f'/gpfs/data/fs71999/uncover_mb/experiments/ws/{stego_method}'
        model_name = diffusion.get_model_name(
            network=NETWORK,
            stego_method=stego_method,
            alpha=alpha,
            drop_rate=drop_rate,
            loss=loss,
        )

        # pixel estimator
        pixel_estimator = diffusion.get_unet_estimator(
            model_path=model_path,
            model_name=model_name,
            # used same image loader
            channels=channels,
        )

        # run WS attack
        res_i = attack(
            input_dir,
            demosaic=None,
            # inbayer=None,
            pixel_estimator=pixel_estimator,
            channels=channels,
            process_image=process_cover,
            imread=imread,
            #
            **kw
        )
        res_i['filter'] = res_i['filter'].apply(lambda s: s + '-' + loss)
        res.append(res_i)
    #
    res = pd.concat(res)
    res['demosaic'] = res['demosaic'].fillna('*')
    res = res[~res.mae.isna()]
    res = res[~res.wmae.isna()]
    print(res)
    res = pd.melt(
        res,
        id_vars=['demosaic', 'filter', 'channels', 'model', 'inbayer', 'information'],
        value_vars=['mae', 'wmae'],
        var_name='metric')
    print(res)

    # #
    # return {
    #     'demosaic': demosaic,
    #     'metric': 'mae',
    #     'filter': 'UNet',
    #     'value': mae,
    #     'channels': ''.join(map(str, channels)),
    #     'model': 'gray',
    #     'inbayer': None,
    #     'information': 'Unconditional'
    # }
    return res


if __name__ == '__main__':
    #
    UNCONDITIONAL = ([None], False)
    CONDITIONAL = (['00', '01', '10', '11'], False)
    CONDITIONAL_ORACLE = (['00', '01', '10', '11'], True)
    NETWORK = 'unet_2'
    # take_num_images = 10
    take_num_images = 250

    # # ALASKA
    # results_dir = pathlib.Path('../results/filters_boss')
    # input_dir = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2024-01-26/')
    # models = ['gray']
    # demosaics = [None]
    # model_to_channels = {'gray': ((3,),)}
    # # BOSS
    # results_dir = pathlib.Path('../results/filters_boss')
    # input_dir = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18/')
    # models = ['gray']
    # demosaics = [None]
    # model_to_channels = {'gray': ((3,),)}

    # color
    results_dir = pathlib.Path('../results/filters_alaska')
    input_dir = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2023-12-18/')
    models = ['gray', 'color8']
    demosaics = ['ahd']
    model_to_channels = {
        'gray': [(0,), (1,), (2,)],
        'color8': [(0, 1, 2), (1, 0, 2), (2, 0, 1)],
    }


    model_to_filters = {
        'gray': [
            ['AVG']*4,
            ['KB']*4,
            # ['OLS']*4,
        ],
        'color8': [
            # ['AVG']*4,
            # ['KB']*4,
            ['OLS']*4,
        ],
    }
    kernel_to_advantage = {
        'AVG': [UNCONDITIONAL],
        'KB': [UNCONDITIONAL],
        'OLS': [UNCONDITIONAL, CONDITIONAL],
    }

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(results_dir / "kb.log"),  # log to log file
            # logging.StreamHandler(),  # print to stderr
        ]
    )

    df_filters = filters.evaluate.evaluate_linear_filters(
        input_dir=input_dir,
        results_dir=results_dir,
        models=models,
        demosaics=demosaics,
        model_to_channels=model_to_channels,
        model_to_filters=model_to_filters,
        kernel_to_advantage=kernel_to_advantage,
        #
        take_num_images=take_num_images,
        split='split_te.csv',
        progress_on=True,
    )

    # DEBUG
    df = df_filters

    # df_unet = evaluate_unet(
    #     input_dir=input_dir,
    #     #
    #     take_num_images=take_num_images,
    #     split='split_te.csv',
    #     progress_on=True,
    # )

    # prediction to latex
    # df = pd.concat([df_filters, df_unet])
    print(df)
    filters.evaluate.output_to_latex(df, results_dir)
