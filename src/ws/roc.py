
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import typing

try:
    from .. import _defs
    from .. import detector
    from .. import diffusion
    from .. import fabrika
    from . import predict
except ImportError:
    sys.path.append('.')
    sys.path.append('ws')
    import _defs
    import detector
    import diffusion
    import fabrika
    import predict


def _attack(
    fname: str,
    demosaic: str,
    detect: typing.Callable = None,
    imread: typing.Callable = None,
    # process_image: typing.Callable = None,
    **kw
) -> float:
    """"""
    # read image
    x = imread(fname)

    # detect
    score = detect(x)

    return {
        'demosaic': demosaic,
        'score': score,
        **kw
    }

@fabrika.precovers(iterator='joblib', ignore_missing=True, n_jobs=4)  # os.cpu_count())
def attack_cover(*args, **kw):
    return _attack(*args, **kw)


@fabrika.stego_spatial(iterator='joblib', ignore_missing=True, n_jobs=4)  # os.cpu_count())
def attack_stego(*args, **kw):
    return _attack(*args, **kw)


def run_boss(
    input_dir: pathlib.Path,
    stego_method: str,
    alpha: float,
    model_name: str,
    model_path: str,
    no_stem_stride: bool = False,
    lsbr_reference: bool = False,
    imread: typing.Callable = _defs.imread4_f32,  # reads image
    **kw,
) -> float:

    # attack configuration
    if stego_method:
        attack = attack_stego
        kw_attack = {
            'stego_method': stego_method,
            'alpha': alpha,
        }
    else:
        attack = attack_cover
        kw_attack = {}

    # detector
    in_channels = 1
    in_channels += int(lsbr_reference)
    detect = detector.get_b0_detector(
        model_path=model_path,
        model_name=model_name,
        in_channels=in_channels,
        shape=(512, 512),
        no_stem_stride=no_stem_stride,
        lsbr_reference=lsbr_reference,
        # channels=c,
    )

    # run WS attack
    res = attack(
        input_dir,
        # demosaic=None,
        inbayer=None,
        **kw_attack,
        detect=detect,
        # process_image=process_cover,
        imread=imread,
        no_stem_stride=no_stem_stride,
        **kw,
    )
    prefix = ''
    if no_stem_stride:
        prefix += 'ns-'
    if lsbr_reference:
        prefix += 'r-'
    res['model_name'] = prefix + 'B0'
    return res


def collect_ws_attacks(
    input_dir: pathlib.Path,
    model_names: typing.List[str],
    stego_methods: typing.List[str],
    alphas: typing.List[float],
    matched: bool = True,
    **kw,
) -> pd.DataFrame:
    model_path = pathlib.Path('/gpfs/data/fs71999/uncover_mb/experiments/ws') / stego_methods[-1]
    res = []
    for stego_method in stego_methods:
        for alpha in alphas if stego_method else [.0]:
            # WS
            for model_name in model_names:
                print(stego_method, alpha, model_name)
                if model_name == 'UNet':
                    model_name = diffusion.get_model_name(
                        stego_method=stego_methods[1] if matched else 'LSBr',
                        alpha=.4,
                    )
                weighted = 0  # if model_name == 'UNet' else 1
                res_i = predict.run_ws(
                    input_dir=input_dir,
                    stego_method=stego_method,
                    alpha=alpha,
                    #
                    # model=model,
                    model_path=model_path,
                    kernel_path=pathlib.Path('../results/filters_boss/gray'),
                    model_name=model_name,
                    weighted=weighted,
                    correct_bias=False,
                    #
                    **kw,
                )
                res.append(res_i)
                # print(res_i)  # .to_string())

    # B0
    for stego_method in stego_methods:
        for alpha in alphas if stego_method else [.0]:
            for train_alpha, no_stem_stride, lsbr_reference in [
                [.01, False, False],
                [.01, True, False],
                # [.1, True, False],
                [.01, True, True]
            ]:
                print(stego_method, alpha, 'B0', train_alpha, no_stem_stride, lsbr_reference)
                model_name = detector.get_model_name(
                    stego_method=stego_methods[1],
                    alpha=train_alpha,
                    no_stem_stride=no_stem_stride,
                    lsbr_reference=lsbr_reference,
                )

                res_i = run_boss(
                    input_dir=input_dir,
                    stego_method=stego_method,
                    alpha=alpha,
                    #
                    # model=model,
                    model_path=model_path,
                    model_name=model_name,
                    no_stem_stride=no_stem_stride,
                    lsbr_reference=lsbr_reference,
                    #
                    **kw
                )
                res_i['model_name'] = res_i['model_name'] + f'_{train_alpha}'
                res.append(res_i)

    res = pd.concat(res).reset_index(drop=True)
    res['stego_method'] = res['stego_method'].fillna('Cover')
    res['alpha'] = res['alpha'].fillna(0.)
    return res


def produce_roc(df_ws: pd.DataFrame) -> pd.DataFrame:
    #
    df = []
    for (stego_method, model_name), _ in df_ws.groupby(['stego_method', 'model_name']):
        if stego_method == 'Cover':
            continue
        # print(stego_method, model_name)

        # filter
        df_ws_i = df_ws[df_ws['model_name'] == model_name]
        df_ws_i = df_ws_i[df_ws_i['stego_method'].isin([stego_method, 'Cover'])]

        # get prediction
        if 'B0' in model_name:
            y_hat = df_ws_i['score'].to_numpy()
            y = df_ws_i['alpha'].to_numpy()
        else:
            y_hat = np.clip(df_ws_i['beta_hat'].to_numpy(), 0, None)
            y = df_ws_i['alpha'].to_numpy() / 2

        # iterate thresholds
        tpr, fpr, taus = [], [], []  # [1.], [1.], [1e-10]
        for tau in reversed(np.linspace(0, 1, 501, endpoint=True)):

            # confusion matrix
            TP = np.sum((y_hat > tau) & (y > 0.))
            FP = np.sum((y_hat > tau) & (y <= 0.))
            TN = np.sum((y_hat <= tau) & (y <= 0.))
            FN = np.sum((y_hat <= tau) & (y > 0.))

            # calculate TPR and FPR
            taus.append(tau)
            tpr.append(TP / (TP + FN))
            fpr.append(FP / (FP + TN))
        tpr, fpr = np.array(tpr), np.array(fpr)
        taus = np.array(taus)
        # #
        # print('FPR:', fpr[:5], '...', fpr[-5:])
        # print('TPR:', tpr[:5], '...', tpr[-5:])
        # print('y_hat:', y_hat[:5], y_hat.min(), y_hat.max())
        # print('y:', y[:5], y.min(), y.max())
        # print('tau:', taus[:5], taus[-5:], taus.min(), taus.max())
        #
        bins = np.diff(fpr, prepend=fpr[0])
        bins /= bins.sum()
        # print('bins:', bins[:5], bins[-5:], bins.sum())
        auc = np.sum(bins * tpr)
        tau0_idx = np.argmin((1 - tpr + fpr)/2)
        p_e = ((1 - tpr + fpr)/2)[tau0_idx]
        # decision at 0.5
        TP = np.sum((y_hat > .5) & (y > 0.))
        FP = np.sum((y_hat > .5) & (y <= 0.))
        TN = np.sum((y_hat <= .5) & (y <= 0.))
        fpr50, tpr50 = FP / (FP + TN), TP / (TP + FN)

        print(
            stego_method, model_name,
            f'P_E={p_e} [{taus[tau0_idx]}]',
            f'AUC={auc}',
            f'err_tau0=[{fpr[tau0_idx], tpr[tau0_idx]}]'
            f'err_tau50=[{fpr50, tpr50}]'
        )

        # plot
        if 'B0' in model_name:
            label = model_name
        else:
            label = f'WS-{model_name}'
        df.append(pd.DataFrame({
            'stego_method': stego_method,
            'model_name': model_name,
            'tau': taus,
            'tpr': tpr,
            'fpr': fpr,
            'p_e': p_e,
            'tau0': taus[tau0_idx],
            'fpr_tau0': fpr[tau0_idx],
            'tpr_tau0': tpr[tau0_idx],
            'auc': auc,
            'fpr_50': fpr50,
            'tpr_50': tpr50,
            'label': label,
        }))

    df = pd.concat(df)
    return df


# if __name__ == '__main__':
#     # ALASKA_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2023-12-18')
#     BOSS_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18')
#     # STEGO_METHODS = [None, 'LSBr']  #, 'HILLr']  # ['LSBr', 'HILLr']
#     # ALPHAS = [.05, .01]  # [.4, .2, .1, .05, .01]  # .01, .05]  #.1, .2]
#     # ALPHAS = [.4, .2, .1]  # [.4, .2, .1, .05, .01]  # .01, .05]  #.1, .2]
#     # MODEL_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/experiments/ws') / STEGO_METHODS[-1]
#     # MODEL_NAMES = ['KB', 'UNet']  # ['AVG', 'KB', 'OLS', 'UNet']  #['AVG', 'KB', 'OLS']  # , 'UNet']
#     take_num_images = 1000

#     for alphas, band in zip(
#         [
#             # [.1, .2],
#             # [.01, .05],
#             [.01],
#             [.05],
#             [.1],
#             [.2],
#         ],
#         [
#             # 'high',
#             # 'low',
#             '0.01',
#             '0.05',
#             '0.1',
#             '0.2',
#         ],
#     ):
#         print(f'=== {band}:{alphas} ===')

#         # run WS
#         df_ws = collect_ws_attacks(
#             input_dir=BOSS_PATH,
#             model_names=['AVG', 'KB', 'OLS', 'UNet'],
#             stego_methods=[None, 'LSBr'],
#             alphas=alphas,
#             matched=False,
#             #
#             split='split_te.csv',
#             take_num_images=take_num_images,
#             progress_on=True,
#         )
#         # df_ws.to_csv(f'../text/img/df_ws_{band}.csv', index=False)

#         # # compute ROC curves
#         # df_ws = pd.read_csv(f'../text/img/df_ws_{band}.csv')
#         # # print(df_ws)
#         df_roc = produce_roc(df_ws=df_ws)

#         # # plot ROC curves
#         # fig, ax = plt.subplots()
#         # for (label), df_roc_i in df_roc.groupby(['label']):
#         #     df_roc_i = df_roc_i.sort_values('tau')
#         #     ax.plot(df_roc_i['fpr'], df_roc_i['tpr'], label=label)
#         # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
#         # ax.set_xlabel('False Positive Rate (FPR)')
#         # ax.set_ylabel('True Positive Rate (TPR)')
#         # ax.legend(loc='lower right')
#         # fig.savefig(f'../text/img/roc_ws_{band}.png', bbox_inches='tight', dpi=600)

#         # export AUC
#         df_auc = df_roc[['stego_method', 'model_name', 'auc', 'p_e', 'tau0', 'fpr_tau0', 'tpr_tau0', 'fpr_50', 'tpr_50']].drop_duplicates()
#         df_auc.to_csv(f'../text/img/auc_ws_{band}_mismatched.csv', index=False)

#         # # export
#         # df = df_roc.pivot(
#         #     index=['tau'],
#         #     columns=['stego_method', 'model_name'],
#         #     values=['tpr', 'fpr'],
#         # )#.sort_values('tau')
#         # # print(df.to_string())
#         # df.columns = ['_'.join(col).strip() for col in df.columns.values]
#         # df.to_csv(f'../text/img/roc_ws_{band}.csv', index=False)


if __name__ == '__main__':
    ALASKA_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2024-01-26')
    # STEGO_METHODS = [None, 'LSBr']  #, 'HILLr']  # ['LSBr', 'HILLr']
    # ALPHAS = [.05, .01]  # [.4, .2, .1, .05, .01]  # .01, .05]  #.1, .2]
    # ALPHAS = [.4, .2, .1]  # [.4, .2, .1, .05, .01]  # .01, .05]  #.1, .2]
    # MODEL_PATH = pathlib.Path('/gpfs/data/fs71999/uncover_mb/experiments/ws') / STEGO_METHODS[-1]
    # MODEL_NAMES = ['KB', 'UNet']  # ['AVG', 'KB', 'OLS', 'UNet']  #['AVG', 'KB', 'OLS']  # , 'UNet']
    take_num_images = 1000

    for alphas, band in zip(
        [
            # [.01],
            # [.05],
            [.1],
            # [.2],
            # [.4],
        ],
        [
            # '0.01',
            # '0.05',
            '0.1',
            # '0.2',
            # '0.4',
        ],
    ):
        print(f'=== {band}:{alphas} ===')

        # run WS
        df_ws = collect_ws_attacks(
            input_dir=ALASKA_PATH,
            demosaic=None,#'linear',#['ahd', 'ppg', 'vng'],
            channels=(3,),
            model_names=['AVG', 'KB', 'UNet'],
            stego_methods=[None, 'LSBr'],
            alphas=alphas,
            matched=False,
            #
            split='split_te.csv',
            take_num_images=take_num_images,
            progress_on=True,
            shuffle_seed=12345,
        )
        # df_ws.to_csv(f'../text/img/df_ws_{band}.csv', index=False)

        # # compute ROC curves
        # df_ws = pd.read_csv(f'../text/img/df_ws_{band}.csv')
        # # print(df_ws)
        df_roc = produce_roc(df_ws=df_ws)

        # plot ROC curves
        fig, ax = plt.subplots()
        for (label), df_roc_i in df_roc.groupby(['label']):
            df_roc_i = df_roc_i.sort_values('tau')
            ax.plot(df_roc_i['fpr'], df_roc_i['tpr'], label=label)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.legend(loc='lower right')
        fig.savefig(f'../text/img/roc_ws_{band}_alaska.png', bbox_inches='tight', dpi=600)

        # export AUC
        df_auc = df_roc[['stego_method', 'model_name', 'auc', 'p_e', 'tau0', 'fpr_tau0', 'tpr_tau0', 'fpr_50', 'tpr_50']].drop_duplicates()
        df_auc.to_csv(f'../text/img/auc_ws_{band}_alaska.csv', index=False)

        # export
        df = df_roc.pivot(
            index=['tau'],
            columns=['stego_method', 'model_name'],
            values=['tpr', 'fpr'],
        )#.sort_values('tau')
        # print(df.to_string())
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df.to_csv(f'../text/img/roc_ws_{band}_alaska.csv', index=False)
