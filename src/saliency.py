

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.signal
import sys
import torch
import typing
sys.path.append('.')
import _defs
import diffusion
import filters


# def plot_contour(
#     fname: Path,
#     x: np.ndarray,
#     d: np.ndarray,
#     model_name: str,
#     vmax: float = None
# ):
#     # print(d.max())
#     # if vmax is None:
#     #     vmax = d.max()
#     #
#     fig, ax = plt.subplots()
#     ax.imshow(np.abs(d), vmin=0, vmax=60, cmap='gray_r', interpolation='nearest')
#     # levels = [1.5, 3.5, 7.5, d.max()]
#     # levels = np.logspace(np.log2(.5), np.log2(vmax), base=2)
#     # levels = np.linspace(0., vmax, 100, endpoint=True)
#     # alphas = np.linspace(0, 1, len(levels)-1, endpoint=True)**(1/2)
#     # ax.contourf(d, alpha=1., levels=levels, cmap='Greens')  # linewidths=.5,
#     outname = Path(f'../text/img/contour_{model_name}_{fname.stem}.png')
#     ax.set_axis_off()
#     fig.savefig(outname, dpi=300, bbox_inches='tight', pad_inches=0)
#     print(f'{model_name} contour saved to {outname.absolute()}')

def get_locations(
    fname: Path,
    *,
    channels: typing.Tuple[int] = (3,),
    imread: typing.Callable = _defs.imread_u8,
    **kw,
):
    # load image
    x = imread(fname)
    Image.fromarray(x[..., 0].astype('uint8')).save('../text/img/saliency_image.png')

    gh = filters.predict.infere_single(
        x,
        np.array([
            [+1, +2, +1],
            [0, 0, 0],
            [-1, -2, -1],
        ], dtype='float32')[..., None]
    )
    gv = filters.predict.infere_single(
        x,
        np.array([
            [+1, 0, -1],
            [+2, 0, -2],
            [+1, 0, -1],
        ], dtype='float32')[..., None]
    )
    g = filters.predict.infere_single(
        np.sqrt(gh**2 + gv**2),
        np.array([
            [+1, +1, +1],
            [+1, +1, +1],
            [+1, +1, +1],
        ], dtype='float32')[..., None]
    )
    # g = np.sqrt(gh**2 + gv**2)
    #
    gh_max = np.unravel_index(np.abs(gh/(.1+gv)).argmax(), gh.shape)
    gv_max = np.unravel_index(np.abs(gv/(.1+gh)).argmax(), gv.shape)
    g_max = np.unravel_index(g.argmax(), g.shape)
    g_min = np.unravel_index(g.argmin(), g.shape)
    print(f'{gh_max=}')
    print(f'{gv_max=}')
    print(f'{g_max=}')
    print(f'{g_min=}')
    x = np.round(x).astype('uint8')
    y = np.repeat(x, 3, axis=-1)
    print(y.min(), y.max())
    y[gh_max[:2]] = [255, 0, 0]
    y[gv_max[:2]] = [255, 0, 0]
    y[g_max[:2]] = [255, 0, 0]
    y[g_min[:2]] = [255, 0, 0]
    Image.fromarray(y).save('../text/img/saliency_image_dots.png')
    # fig, ax = plt.subplots()
    # ax.imshow(y)
    # fig.savefig('changes.png', dpi=600, bbox_inches='tight')


def get_unet_saliency(
    fname: Path,
    i: int, j: int,
    *,
    loss: str = 'l1',
    stego_method: str = 'LSBr',
    channels: typing.Tuple[int] = (3,),
    device: torch.nn.Module = torch.device('cpu'),
    imread: typing.Callable = _defs.imread_u8,
    **kw,
):
    # load model
    device = torch.device('cpu')
    if loss == 'l1ws':
        model_name = diffusion.get_model_name(
            network='unet_2',
            stego_method=stego_method,
            alpha=.4,
            drop_rate=.0,
            loss='l1ws',
        )
        model_path = Path('/gpfs/data/fs71999/uncover_mb/experiments/ws/') / stego_method
    else:
        stego_method = None
        model_name = diffusion.get_model_name(
            stego_method='dropout',
            alpha=None,
            drop_rate=.1,
            loss='l1',
        )
        model_path = Path('/gpfs/data/fs71999/uncover_mb/experiments/ws/dropout')
    model = diffusion.get_pretrained(
        model_path=model_path,
        channels=channels,
        device=device,
        model_name=model_name,
        **kw,
    )
    model.input_dropout = None

    # load image
    x = imread(fname)

    # evaluation mode, requires gradient
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # transform input
    transform = diffusion.data.get_timm_transform(
        mean=None,
        std=None,
        grayscale=True,
        demosaic_oracle=False,
        post_flip=False,
        post_rotate=False,
    )
    x_ = transform(x / 255.)[None]
    x_.requires_grad = True
    x_ = x_.to(device)
    # infere
    xhat = model(x_)
    # xhat = diffusion.infere_single(x, model=model, device=device)

    # Get the index corresponding to the maximum score and the maximum score itself.
    # score, indices = torch.max(xhat, 1)
    pixel = xhat[0, 0, i, j]

    # get gradients
    pixel.backward()

    # saliency map in 15x15
    slc = x_.grad.data
    # slc = x_.grad.data.abs()
    # slc = (slc - slc.min())/(slc.max() - slc.min())
    n = 8

    stego_method = f'_{stego_method}' if stego_method else ''
    with open(f'../text/img/saliency_{loss}{stego_method}_{i}-{j}.csv', 'w') as fp:
        for dx in range(-n, n+1):
            for dy in range(-n, n+1):
                v = slc[0, 0, i+dx, j+dy].detach().numpy()
                fp.write(f'{dx}\t{dy}\t{v}\n')

    # slcN = slc[0, 0, i-n:i+n+1, j-n:j+n+1].detach().numpy()
    # print(slcN)
    return



    fig, ax = plt.subplots(1, 1)
    if loss == 'l1':
        vmin, vmax = -1, 1
    else:
        vmin, vmax = -.5, .5
    im = ax.imshow(slcN, vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # ax.xticks([])
    # ax.yticks([])
    fig.savefig(f'saliency_{loss}_{i}-{j}.png', dpi=600, bbox_inches='tight')

    # # difference image
    # x = x[1:-1, 1:-1]
    # d = x[..., 0] - xhat[..., 0]
    # print('U-Net MAE:', np.mean(np.abs(d)))

    # return d


if __name__ == '__main__':
    BOSS_PATH = Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18/images/')
    fname = BOSS_PATH / '6.png'
    imread = _defs.imread_f32
    get_locations(
        fname,
        imread=imread,
    )

    for i, j in [
        (307, 10),  # gh_max
        (261, 64),  # gv_max
        (155, 381),  # g_max
        (9, 25),  # g_min
        # # others
        # (100, 100),
        # (200, 200),
        # (300, 300),
        # (400, 400),
    ]:
        #
        sal_unet = get_unet_saliency(
            fname,
            i=i,
            j=j,
            imread=imread,
            loss='l1',
            stego_method='HILLr'
        )
    # d_kb = get_filter_difference(
    #     fname,
    #     imread=imread,
    #     model_name='KB',
    # )
