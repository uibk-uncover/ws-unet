"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import pandas as pd
import pathlib
import sys
sys.path.append('.')
import _defs
import diffusion
import ws


if __name__ == '__main__':
    # ALASKA_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2023-12-18')
    BOSS_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18')
    STEGO_METHODS = [None, 'HILLr']  # None, 'LSBr', 'HILLr'
    ALPHAS = [.1, .2, .4]
    NETWORKS = ['unet_0', 'unet_1', 'unet_2', 'unet_3', 'unet_4']
    model = 'gray'
    matched = False
    take_num_images = 250
    # take_num_images = None

    # images
    res = []
    for network in NETWORKS:
        for stego_method in STEGO_METHODS:
            for alpha in ALPHAS if stego_method else [.0]:
                print(network, stego_method, alpha)

                # models
                model_stego_method = STEGO_METHODS[-1] if matched else 'LSBr'
                model_path = f'/gpfs/data/fs71999/uncover_mb/experiments/ws/{model_stego_method}'
                model_name = diffusion.get_model_name(
                    network=network,
                    stego_method=model_stego_method,
                    alpha=.4,
                    drop_rate=.0,
                    loss='l1ws',
                )
                res_i = ws.predict.run_boss(
                    input_dir=BOSS_PATH,
                    stego_method=stego_method,
                    alpha=alpha,
                    #
                    model=model,
                    model_path=model_path,
                    kernel_path=pathlib.Path(f'../results/filters_boss/{model}'),
                    model_name=model_name,
                    correct_bias=False,
                    weighted=0,
                    #
                    split='split_te.csv',
                    take_num_images=take_num_images,
                    progress_on=True,
                )
                res_i['model_name'] = f'{network}_L1ws'
                res.append(res_i)
    #
    res = pd.concat(res).reset_index(drop=True)
    if 'stego_method' in res:
        res['stego_method'] = res['stego_method'].fillna('Cover')
    else:
        res['stego_method'] = 'Cover'
    res['alpha'] = res['alpha'].fillna(0.)
    res['alpha'] = res['alpha'].apply(lambda a: f'{a:.2f}')
    matched_s = '' if matched else '_mismatch'
    # print(res)
    print(
        res
        .groupby(by=['model_name', 'stego_method', 'alpha'])
        .agg({'beta_hat': [np.median, np.mean]})
    )

    res_q = (
        res
        .groupby(['stego_method', 'alpha', 'model_name'])
        .agg({'beta_hat': [
            'min',
            _defs.iqr_interval(.25, sign=-1.5),
            _defs.quantile(.25),
            _defs.quantile(.5),
            _defs.quantile(.75),
            _defs.iqr_interval(.75, sign=1.5),
            'max',
        ]})
        .reset_index(drop=False)
    )
    res_q.columns = ['_'.join(map(str, col)).strip() for col in res_q.columns.values]
    res_q.to_csv(f'../text/img/grid_k_{STEGO_METHODS[1]}{matched_s}_boxes.csv', index=False)
    print(res_q)

    #
    res['stem'] = res['name'].apply(lambda f: pathlib.Path(f).stem)
    res = res.pivot(index='stem', columns=('stego_method', 'alpha', 'model_name'), values=('beta_hat'))
    res = res.reset_index(drop=True)

    res.columns = [
        '_'.join([
            str(c)
            for c in col
            if c
        ]).strip()
        for col in res.columns.values
    ]
    res.to_csv(f'../text/img/grid_k_{STEGO_METHODS[1]}{matched_s}.csv', index=False)
    print(res)
