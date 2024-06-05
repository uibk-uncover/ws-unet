
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import scipy
import seaborn as sns
import stegolab2 as sl2
import torch
from torchinfo import summary
import torchvision.transforms as transforms

import diffusion
#from diffusion.model import get_model


def get_linear_predictor(model):
    def predict(x):
        return scipy.signal.convolve(x, model, mode='valid')[..., :1]
    # print(f'=> KB filter: {model.shape}')
    return predict


def get_torch_predictor(model):
    def predict(x):
        x_ = transforms.ToTensor()(x)[None].to(DEVICE)
        y_ = model(x_)
        y = y_.detach().numpy()[0, :1, 1:-1, 1:-1]
        return y
    # print('=> UNet predictor')
    # summary(model, (1, 3, 512, 512))
    return predict


def run(demosaic: str, alpha_hat: float, predictor: str):

    data_path = Path(f'/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2023-12-18/images_ahd_smart')
    # data_path = Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18/images')

    if predictor == 'KB':
        predict = get_linear_predictor(
            np.array([[-1, 2, -1], [2, 0, 2], [-1, 2, -1]])[..., None] / 4.
        )
    elif predictor == 'AVG':
        predict = get_linear_predictor(
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])[..., None] / 8.
        )
    elif predictor == '1':
        predict = get_linear_predictor(
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])[..., None] / 1.
        )
    elif predictor == 'UNet':
        model_name = diffusion.get_model_name(
            network='unet_2',
            stego_method='LSBr',
            alpha=.4,
            drop_rate=.0,
            loss='l1ws',
        )
        model = diffusion.get_pretrained(
            MODEL_PATH,
            channels=(3,),
            model_name=model_name,
            device=DEVICE
        )
        predict = lambda x: diffusion.infere_single(x, model=model, device=DEVICE)

    elif predictor == 'cnn':
        model = diffusion.get_pretrained(
            MODEL_PATH,
            channels=(3,),
            network='cnn',
            # model_name='240216153104-2784836-unet_0-alpha_0.400_grayscale_ws_lr_0.0001_',
            device=DEVICE,
        )
        predict = lambda x: diffusion.infere_single(x, model=model, device=DEVICE)

    else:
        raise NotImplementedError(f'unknown predictor: {predictor}')

    # get input
    # for fname in ['56373.png', '06837.png']:
    for fname in ['41550.png', '40846.png']:
    # for fname in ['6104.png', '9168.png', '1137.png', '4735.png']:
        # load cover
        file_path = data_path / fname
        x_bgr = cv2.imread(str(file_path))
        xc = cv2.cvtColor(x_bgr, cv2.COLOR_BGR2GRAY)[..., None]
        # x = np.concatenate([x_bgr[..., ::-1], x_y], axis=-1)
        # xc = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)[..., None]
        # xc = np.array(Image.open(file_path))
        # xc = np.array(Image.open(file_path))[..., None]

        # label
        xc_r = xc[1:-1, 1:-1, :1].astype('float32')

        # embed LSBr
        xs = xc + sl2.lsb.simulate(xc, alpha=alpha_hat, seed=12345)

        # prepare input
        # x = xs if use_stego else xc
        x = xs.astype('float32')
        # print(f'=> loaded input: {x.shape}')

        # forward pass
        yc_r = predict(x)
        # print(f'=> predicted output: {y.shape}')

        # residual
        resid = xc_r - yc_r
        print(yc_r.shape, xc_r.shape)
        resid_abs = np.abs(resid)

        # errors
        mae = resid_abs.mean()
        rho = sl2.hill.compute_rho(xc_r[:, :, 0])[:, :, None]
        rho[np.isinf(rho) | np.isnan(rho) | (rho > 10**10)] = 10**10
        wmae = resid_abs[rho <= np.quantile(rho, .1)].mean()
        print(f' - {fname=}\t{mae=}\t{wmae=}')

        # save
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        #
        sns.heatmap(xc_r[:, :, 0], vmin=0, vmax=255, square=True, ax=ax[0])
        ax[0].set_title(f'Cover image [{demosaic}]')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        #
        sns.heatmap(yc_r[:, :, 0], vmin=0, vmax=255, square=True, ax=ax[1])
        ax[1].set_title('Predicted image')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        #
        sns.heatmap(resid[:, :, 0], vmin=0, vmax=80, square=True, cmap='rocket_r', ax=ax[2])
        ax[2].set_title(f'Prediction residual [MAE={mae:.4f}, wMAE={wmae:.4f}]')
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        #
        sns.histplot(
            data=pd.DataFrame({'x': resid_abs.flatten()}),
            x='x', color='orange', bins=61*4+1, ax=ax[3])
        ax[3].set_title('Residual distribution')
        ax[3].set_xlabel(None)
        ax[3].set_ylabel(None)
        ax[3].get_yaxis().set_visible(False)
        ax[3].set_xlim(-1, 60)
        # save
        demosaic_s = f'_demosaic_{demosaic}' if demosaic is not None else ''
        fig.savefig(f'../results/diffusion/{predictor}{demosaic_s}_alpha_{alpha_hat:.2f}_{Path(fname).stem}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    #
    MODEL_PATH = Path('models/dropout')
    MODEL_PATH = Path('models/LSBr')
    DEVICE = torch.device('cpu')
    #
    DEMOSAIC = [None]  # 'linear', 'ahd']
    ALPHA_HAT = [0, .2, 1.]
    # PREDICTOR = ['1', 'AVG', 'KB', 'UNet']
    PREDICTOR = ['UNet']

    #
    for demosaic in DEMOSAIC:
        for predictor in PREDICTOR:
            for alpha_hat in ALPHA_HAT:
                print(f'{demosaic=} {alpha_hat=} {predictor=}')
                run(
                    demosaic=demosaic,
                    alpha_hat=alpha_hat,
                    predictor=predictor,
                )