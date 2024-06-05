
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import scipy.signal
import sys
import typing
sys.path.append('.')
import _defs
import diffusion
import fabrika
import filters
# import predict


@fabrika.cover_stego_spatial(iterator='joblib', convert_to='pandas', ignore_missing=True, n_jobs=50)
def run(
    fname: pathlib.Path,
    name_c: str,
    name_s: str,
    predictor: typing.Callable,
    **kw,
):
    # get cover and stego pair
    dataset = fname.parents[len(pathlib.Path(name_c).parents) - 1]
    path_c = dataset / name_c
    path_s = dataset / name_s

    # get change mask
    x_c = _defs.imread_f32(path_c)  #/ 255.
    x_s = _defs.imread_f32(path_s)  #/ 255.
    d_s = (x_s - x_c)[1:-1, 1:-1]
    # print(x_c[1:-1, 1:-1][:3, :3, 0])
    # print(x_s[1:-1, 1:-1][:3, :3, 0])
    # print(d_s[1:-1, 1:-1][:3, :3, 0])

    # estimate cover from stego
    xhat_c = predictor(x_s)
    dhat_c = xhat_c - x_c[1:-1, 1:-1]

    # measure correlation of cover estimate with change mask
    # rho = scipy.signal.correlate(xhat_c / 255., d_s[1:-1, 1:-1] / 255., mode='valid')[0, 0, 0]
    # rho = scipy.signal.correlate(dhat_s, d_s, mode='valid')[0, 0, 0]
    # print(dhat_s.shape, d_s.shape)

    # dhat_s_norm = (dhat_s - dhat_s.mean()) / dhat_s.std()
    # d_s_norm = (d_s - d_s.mean()) / d_s.std()
    # cor = np.sum(dhat_s_norm * d_s_norm) / (d_s.size - 1)
    cov = np.sum((xhat_c - xhat_c.mean())*(d_s - d_s.mean()))/(d_s.size - 1)
    cor = cov / xhat_c.std() / d_s.std()

    # measure significance
    test_val = np.abs(cor)/np.sqrt(1-cor**2)*np.sqrt(d_s.size - 2)
    pval = scipy.stats.t.sf(test_val, d_s.size - 2)

    return {
        'name_c': str(name_c),
        'name_s': str(name_s),
        'cor': cor,
        'pval': pval,
        'std_d_s': d_s.std(),
        'std_dhat_c': xhat_c.std(),
    }


if __name__ == '__main__':
    BOSS_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18')
    FILTER_PATH = pathlib.Path('../results/filters_boss/gray')
    MODEL_NAMES = ['1', 'AVG9', 'AVG', 'KB']
    kw = {'take_num_images': None, 'shuffle_seed': 12345, 'split': 'split_te.csv', 'progress_on': True}

    res = []
    for model_name in MODEL_NAMES:
        print(f'Running {model_name} ...')

        # get predictor
        predictor = filters.get_filter_estimator(
            model_path=FILTER_PATH,
            channels=(3,),
            model_name=model_name,
        )

        # compute correlation
        res_m = run(
            BOSS_PATH,
            stego_method='LSBr',
            alpha=1.,
            predictor=predictor,
            **kw,
        )
        res_m['model_name'] = model_name
        res.append(res_m)

    # get predictor
    for model in ['UNet_L1', 'UNet_L1ws']:
        print(f'Running {model} ...')
        if model == 'UNet_L1':
            model_path = pathlib.Path('/gpfs/data/fs71999/uncover_mb/experiments/ws/dropout')
            model_name = diffusion.get_model_name(
                stego_method='dropout',
                alpha=None,
                drop_rate=.1,
                loss='l1',
            )
        elif model == 'UNet_L1ws':
            model_path = pathlib.Path('/gpfs/data/fs71999/uncover_mb/experiments/ws/LSBr')
            model_name = diffusion.get_model_name(
                stego_method='LSBr',
                alpha=.4,
                # stego_method=STEGO_METHODS[1],
                # alpha=ALPHAS[0],
                drop_rate=.0,
                loss='l1ws',
            )
        predictor = diffusion.get_unet_estimator(
            model_path=model_path,
            model_name=model_name,
            channels=(3,),
        )
        # compute correlation
        res_m = run(
            BOSS_PATH,
            stego_method='LSBr',
            alpha=1.,
            predictor=predictor,
            **kw,
        )
        res_m['model_name'] = model
        res.append(res_m)

    res = pd.concat(res).reset_index(drop=True)
    model_names = res.model_name.unique().tolist()
    res = res.groupby('model_name').agg({'cor': ['mean', 'median'], 'pval': ['mean', 'median']})  #.reset_index(drop=False)
    print(res)
    res = res[['cor', 'pval']].T
    # res = res.pivot(columns='model_name', values='cor')
    res = res[model_names].reset_index(drop=False)
    # res.name_c = res.name_c.apply(lambda f: pathlib.Path(f).name)

    t = res.to_latex(
        index=False,
        # header=False,
        float_format=lambda i: f'${i:.04f}$',
        # index_names=False,
        escape=False,
    )
    print(t)
