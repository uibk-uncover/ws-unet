"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import sys
import torch
import typing

sys.path.append('.')
import _defs
import fabrika
import filters
import unet

#
NUM_PIXELS = None
NUM_IMAGES = None
DATA_PATH = '../data'
DEVICE = torch.device('cpu')


def subset_residual(resid: np.ndarray, fname: str, size: int):
    """"""
    # convert name to seed
    seed = fabrika.filename_to_image_seed(fname)
    rng = np.random.default_rng(seed)

    # subset residual
    if size:
        selected = rng.integers(resid.size, size=size)
        selected = (selected // resid.shape[1], selected % resid.shape[1])
        return resid[selected]
    else:
        return resid.flatten()


@fabrika.precovers(iterator='python', convert_to='numpy', ignore_missing=True, n_jobs=50)  # n_jobs=os.cpu_count())
def filter_residuals(
    fname: str,
    **kw,
) -> np.ndarray:
    """"""
    #
    resid = filters.predict.get_filter_residuals(fname=fname, **kw)

    #
    return subset_residual(resid, fname=fname, size=NUM_PIXELS)


def filter_mae(
    model: str,
    channels: typing.Tuple[int],
    model_name: str,
) -> typing.Dict[str, np.ndarray]:
    """"""

    # channels
    channels_name = ''.join(map(str, channels))

    # Image processor
    process_image = _defs.get_processor(
        # model=model,
        channels=channels,
    )

    # get filter
    filter = filters.get_coefficients(
        # model=model,
        model_name=model_name,
        flatten=True,
    )

    # predict
    resid = filter_residuals(
        DATA_PATH,
        demosaic=None,
        inbayer=None,
        filter=filter,
        model_name=model_name,
        channels=channels,
        process_image=process_image,
        #
        take_num_images=NUM_IMAGES,
        split='split_te.csv',
        shuffle_seed=12345,
        progress_on=True,
    )
    return {f'{model_name}_{channels_name}': np.abs(resid)}


@fabrika.precovers(iterator='python', convert_to=None, ignore_missing=True, n_jobs=50)
def unet_residuals(
    fname: str,
    model: typing.Callable,
    channels: typing.Tuple[int],
    imread: typing.Callable = _defs.imread4_f32,
    **kw,
) -> typing.Tuple:
    """"""
    # read image
    x = imread(fname)[..., channels]

    # predict
    y = unet.infere_single(x, model=model, device=DEVICE)
    resid = x[1:-1, 1:-1, :1] - y

    #
    return subset_residual(resid, fname=fname, size=NUM_PIXELS)


def unet_mae(
    channels,
    model_name,
    model_path,
):
    """"""

    # get model
    model = unet.get_pretrained(
        model_path=model_path,
        channels=channels,
        model_name=model_name
    )

    # get
    res = unet_residuals(
        DATA_PATH,
        model=model,
        channels=channels,
        take_num_images=NUM_IMAGES,
        split='split_te.csv',
        shuffle_seed=12345,
        progress_on=True,
    )

    # mean absolute error
    res = np.array([np.abs(resid).flatten() for resid in res])

    # result
    channels_name = ''.join(map(str, channels))
    return {f'UNet_{channels_name}': res}


class SquareRootScale(mpl.scale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kw):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mpl.scale.ScaleBase.__init__(self, axis, **kw)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mpl.ticker.AutoLocator())
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_minor_locator(mpl.ticker.NullLocator())
        axis.set_minor_formatter(mpl.ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class SquareRootTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mpl.scale.register_scale(SquareRootScale)


def plot_error(results, anchor_channel, fname):
    """"""

    # select random pixels
    points = collections.OrderedDict([
        (k, x.flatten())
        for k, x in results.items()
    ])

    # order using anchor
    order = np.argsort(points[anchor_channel])
    points = collections.OrderedDict([
        (k, x.flatten()[order])
        for k, x in results.items()
    ])

    # find edges
    anchor_edge_values = [.5, 1.5, 3.5, 7.5]#, 15.5]
    anchor_edges = [np.argmin(points[anchor_channel] <= e)-1 for e in anchor_edge_values]
    anchor_edges = [0] + anchor_edges + [len(points[anchor_channel])]
    anchor_edge_values = [0] + anchor_edge_values + [np.inf]

    # split along edges
    df = []
    for i, (k, x) in enumerate(points.items()):
        df_i = []
        for j in range(len(anchor_edges)-1):
            df_i.append({
                'Type': k,
                'edge_interval': f'{anchor_edge_values[j]}-{anchor_edge_values[j+1]}',
                'values': x[anchor_edges[j]:anchor_edges[j+1]]
            })
        df_i = pd.DataFrame(df_i).explode('values')
        df.append(df_i)
    df = pd.concat(df)
    # df = pd.DataFrame(df).explode('values').reset_index(drop=True)
    df['values'] = df['values'].astype('float64')
    print(df.groupby(['edge_interval', 'Type']).count())

    fig, ax = plt.subplots()
    sns.boxplot(
        df, x='edge_interval', y='values', hue='Type',
        flierprops={'marker': 'x', 'markerfacecolor': 'black', 'alpha': .1},
        ax=ax,
    )
    ax.set_ylim(0, 64)
    ax.set_yscale('squareroot')
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 1, 4, 9, 16, 25, 36, 49, 64]))
    ax.set_xlabel('Pixels at given AE of KB_gray filter')
    ax.set_ylabel('Absolute Error (AE)')
    outfile = Path(f'../results/prediction/{fname}.png')
    fig.savefig(outfile, dpi=600, bbox_inches='tight')
    print(f'Output saved to {outfile.absolute()}')
    #
    df = (
        df.groupby(['Type', 'edge_interval'])
        .agg({'values': [
            'min',
            _defs.iqr_interval(.25, sign=-1.5),
            _defs.quantile(.25),
            _defs.quantile(.5),
            _defs.quantile(.75),
            _defs.iqr_interval(.75, sign=1.5),
            'max',
        ]})
    )
    df.columns = [col[1] for col in df.columns.values]
    df = df.reset_index().sort_values(['edge_interval', 'Type'])
    df.to_csv(f'../results/prediction/{fname}.csv', index=False)


if __name__ == '__main__':
    #
    results = collections.OrderedDict()
    results |= filter_mae('gray', (3,), 'KB')
    results['KB'] = results.pop('KB_3')

    print('Plotting 3:')
    results_3 = results.copy()
    results_3 |= filter_mae('gray', (3,), 'AVG')
    # results_3['KB'] = results_3.pop('KB')
    results_3['AVG'] = results_3.pop('AVG_3')
    # results_3['OLS'] = results_3.pop('OLS')
    model_name = unet.get_model_name(
        stego_method='dropout',
        alpha=None,
        drop_rate=.1,
        loss='l1',
    )
    model_path = Path('../models/unet/dropout')
    results_3 |= unet_mae(
        (3,),
        model_name=model_name,
        model_path=model_path,
    )
    results_3['UNet_l1'] = results_3.pop('UNet_3')
    model_name = unet.get_model_name(
        stego_method='LSBr',
        alpha=.4,
        drop_rate=.0,
        loss='l1ws',
    )
    model_path = Path('../models/unet/LSBr')
    results_3 |= unet_mae(
        (3,),
        model_name=model_name,
        model_path=model_path,
    )
    results_3['UNet_l1ws'] = results_3.pop('UNet_3')
    plot_error(results_3, 'KB', 'ae_boxes_3')
