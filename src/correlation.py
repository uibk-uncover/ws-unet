"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import pandas as pd
import pathlib
import scipy.signal
import sys
import typing
sys.path.append('.')
import _defs
import fabrika
import filters
import unet



@fabrika.cover_stego_spatial(iterator='python', convert_to='pandas', ignore_missing=True, n_jobs=50)
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
    x_c = _defs.imread_f32(path_c)
    x_s = _defs.imread_f32(path_s)
    d_s = (x_s - x_c)[1:-1, 1:-1]

    # estimate cover from stego
    xhat_c = predictor(x_s)
    dhat_c = xhat_c - x_c[1:-1, 1:-1]

    # measure correlation
    cov = np.sum((dhat_c - dhat_c.mean())*(d_s - d_s.mean()))/(d_s.size - 1)
    cor = cov / xhat_c.std() / d_s.std()

    # measure significance
    test_val = np.abs(cor)/np.sqrt(1-cor**2)*np.sqrt(d_s.size - 2)
    pval = scipy.stats.t.sf(test_val, d_s.size - 2)

    return {
        'name_c': str(name_c),
        'name_s': str(name_s),
        'correlation': cor,
        'p-value': pval,
    }


if __name__ == '__main__':
    #
    DATA_DIR = pathlib.Path('../data')
    MODEL_DIR = pathlib.Path('../models/unet')
    #
    res = []
    for model_name in ['1', 'AVG9', 'AVG', 'KB']:
        print(f'Running {model_name} ...')

        # get predictor
        predictor = filters.get_filter_estimator(filter_name=model_name, flatten=False)

        # compute correlation
        res_m = run(
            DATA_DIR,
            stego_method='LSBR',
            alpha=1.,
            predictor=predictor,
            progress_on=True,
        )
        res_m['model_name'] = model_name
        res.append(res_m)

    # get predictor
    for stego_method in ['dropout', 'LSBR', 'HILLR']:
        model_path = pathlib.Path('../models/unet/') / stego_method
        model_name = unet.get_model_name(stego_method=stego_method)
        config = unet.get_model_config(
            model_dir=MODEL_DIR,
            stego_method=stego_method,
            model_name=model_name,
        )
        #
        predictor = unet.get_unet_estimator(
            model_path=model_path,
            model_name=model_name,
            channels=(3,),
        )
        # compute correlation
        res_m = run(
            DATA_DIR,
            stego_method='LSBR',
            alpha=1.,
            predictor=predictor,
            progress_on=True,
        )
        res_m['model_name'] = f'UNet_{stego_method}_{config["loss"]}'
        res.append(res_m)

    #
    res = pd.concat(res).reset_index(drop=True)
    # res.to_csv('../results/estimation/correlation_individual.csv', index=False)
    model_names = res.model_name.unique().tolist()
    res = res.groupby('model_name').agg({
        'correlation': 'median',
        'p-value': 'median'
    })
    res.T[model_names].to_csv('../results/estimation/correlation.csv')
