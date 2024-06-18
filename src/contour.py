
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.signal
import sys
import torch
import typing
sys.path.append('.')
import _defs
import unet
import filters


def plot_contour(
    fname: Path,
    x: np.ndarray,
    d: np.ndarray,
    model_name: str,
    vmax: float = None
):
    # print(d.max())
    # if vmax is None:
    #     vmax = d.max()
    #
    fig, ax = plt.subplots()
    ax.imshow(np.abs(d), vmin=0, vmax=60, cmap='gray_r', interpolation='nearest')
    # levels = [1.5, 3.5, 7.5, d.max()]
    # levels = np.logspace(np.log2(.5), np.log2(vmax), base=2)
    # levels = np.linspace(0., vmax, 100, endpoint=True)
    # alphas = np.linspace(0, 1, len(levels)-1, endpoint=True)**(1/2)
    # ax.contourf(d, alpha=1., levels=levels, cmap='Greens')  # linewidths=.5,
    outname = Path(f'../results/prediction/contour_{model_name}_{fname.stem}.png')
    ax.set_axis_off()
    fig.savefig(outname, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f'{model_name} contour saved to {outname.absolute()}')


def get_unet_difference(
    fname: Path,
    *,
    model_path: Path = Path('../models/unet/LSBR'),
    channels: typing.Tuple[int] = (3,),
    device: torch.nn.Module = torch.device('cpu'),
    imread: typing.Callable = _defs.imread_u8,
    **kw,
):
    # load model
    device = torch.device('cpu')
    model_name = unet.get_model_name(
        # network='unet_2',
        stego_method='LSBR',
        # alpha=.4,
        # drop_rate=.0,
        # loss='l1ws',
    )
    model = unet.get_pretrained(
        model_path=model_path,
        channels=channels,
        device=device,
        model_name=model_name,
        **kw,
    )

    # load image
    x = imread(fname)

    # infere single image
    xhat = unet.infere_single(x, model=model, device=device)

    # difference image
    x = x[1:-1, 1:-1]
    d = x[..., 0] - xhat[..., 0]
    print('U-Net MAE:', np.mean(np.abs(d)))

    return d


def get_filter_difference(
    fname: Path,
    *,
    # channels: typing.Tuple[int] = (3,),
    model_name: str = None,
    imread: typing.Callable = _defs.imread_f32,
    **kw,
):
    # load model
    model = filters.get_coefficients(
        # channels=channels,
        filter_name=model_name,
        flatten=False,
    )

    # load image
    x = imread(fname)

    # infere single image
    xhat = filters.infere_single(x, model)

    # difference image
    x = x[1:-1, 1:-1]
    d = x[..., 0] - xhat[..., 0]
    print(f'{model_name} MAE:', np.mean(np.abs(d)))

    return d


if __name__ == '__main__':
    COVER_PATH = Path('../data/images')
    fname = COVER_PATH / '6.png'
    imread = _defs.imread_f32

    #
    d_unet = get_unet_difference(
        fname,
        imread=imread,
    )
    d_kb = get_filter_difference(
        fname,
        imread=imread,
        model_name='KB',
    )

    #
    x = imread(fname)
    vmax = max([d_unet.max(), d_kb.max()])
    plot_contour(fname, x, d_unet, 'unet', vmax=vmax)
    plot_contour(fname, x, d_kb, 'KB', vmax=vmax)






# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import seaborn as sns
# import stegolab2 as sl2


# def rolling_window(a, window):
#     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#     strides = a.strides + (a.strides[-1],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# def plot_line_profile(y1: np.ndarray, y2: np.ndarray, window: int = 100):
#     """"""
#     # convert to 1D
#     y1, y2 = y1.flatten(), y2.flatten()

#     # order based on the first
#     idx1 = np.argsort(y1)
#     y1, y2 = y1[idx1], y2[idx1]

#     # smooth the second one using rolling mean/sd
#     y2_windowed = rolling_window(y2, window)
#     y2_mean = y2_windowed.mean(-1)
#     y2_std = y2_windowed.std(-1)

#     # === plot ===
#     fig, ax = plt.subplots()
#     # y2 area
#     ax.fill_between(
#         np.arange(len(y2_mean)),
#         y2_mean-1.96*y2_std,
#         y2_mean+1.96*y2_std,
#         color='orange', alpha=.2, linewidth=0,
#     )
#     ax.plot(np.arange(len(y2_mean)), y2_mean, color='orange')
#     # add y1 line
#     ax.plot(np.arange(len(y1)), y1, color='blue')
#     fig.savefig('mae_profile.png', dpi=300, bbox_inches='tight')


# def run_linear_gray_color():
#     import glob
#     import json

#     dir_name = '/Users/martin/UIBK/fabrika/alaska_20240105/images_ppg/'
#     with open('../../results/filters_alaska/gray/kernels.json') as fp:
#         kernel = json.load(fp)['OLS_0']
#     for fname in glob.glob(f'{dir_name}/*.png')[:10]:
#         print(fname)


# # gray / color
# if __name__ == '__main__':
#     run_linear_gray_color()

# TOY
# if __name__ == '__main__':
#     # load image
#     fname = '/Users/martin/UIBK/fabrika/alaska_20240105/images_ahd/02808.png'
#     x = np.array(Image.open(fname))[..., 0]

#     # compare HILL and S-UNIWARD costs
#     rho1 = sl2.hill.compute_rho(x)
#     rho2 = sl2.suniward.compute_rho(x)

#     # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     # sns.heatmap(rho1, ax=ax[0])
#     # sns.heatmap(rho2, ax=ax[1])
#     # plt.show()

#     # standardize
#     rho1 = (rho1 - rho1.mean()) / rho1.std()
#     rho2 = (rho2 - rho2.mean()) / rho2.std()

#     # create line profile plot
#     plot_line_profile(rho1, rho2)

