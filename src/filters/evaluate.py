
import logging
import os
import pandas as pd
import pathlib
import sys
from typing import List, Dict, Tuple

# import filters
try:
    from . import predict
except ImportError:
    sys.path.append('.')
    sys.path.append('filters')
    import predict


def aggregate_df(x: pd.DataFrame, cols) -> pd.DataFrame:
    return (x.groupby(cols).agg({'value': 'mean'}).reset_index(drop=False))


def evaluate_linear_filters(
    input_dir: pathlib.Path,
    results_dir: pathlib.Path,
    models: List[str],
    demosaics: List[str],
    model_to_channels: Dict[str, List[List[int]]],
    model_to_filters: Dict[str, List[List[str]]],
    kernel_to_advantage: Dict[str, List[Tuple[List[str], bool]]],
    **kw
) -> pd.DataFrame:

    df = []
    for model in models:
        for kernel_names in model_to_filters[model]:
            for demosaic in demosaics:
                for (kernel_inbayers, has_oracle) in kernel_to_advantage[kernel_names[0]]:
                    for inbayer in kernel_inbayers:
                        print(
                            f'Predicting {model} with kernel {kernel_names[0]} '+
                            (f"with oracle [{demosaic}] " if has_oracle else "") +
                            f'for {f"mode {inbayer}," if inbayer else ""}' +
                            (f"demosaic {demosaic}" if demosaic else "")
                        )
                        #
                        df_f = predict.run(
                            input_dir,
                            demosaic=demosaic,
                            model=model,
                            kernel_path=results_dir / model,
                            kernel_names=kernel_names,
                            channels=model_to_channels[model],
                            inbayer=inbayer,
                            has_oracle=has_oracle,
                            **kw
                        )
                        mae_cols = [
                            c
                            for c in df_f.columns
                            if any(c.startswith(prefix) for prefix in {'wmae', 'mae', 'msd'})
                        ]
                        # print(df_f)
                        df_f = df_f.melt(id_vars=['fname', 'demosaic'], value_vars=mae_cols, var_name='metric')
                        df_f['filter'] = df_f.metric.apply(lambda k: '_'.join(k.split('_')[2:]))
                        df_f['channels'] = df_f.metric.apply(lambda k: k.split('_')[1])
                        df_f['metric'] = df_f.metric.apply(lambda k: k.split('_')[0])
                        df_f['demosaic'] = df_f['demosaic'].fillna('*')
                        df_f = pd.concat([
                            aggregate_df(df_f, ['demosaic', 'metric', 'filter'] if kernel_names else ['demosaic', 'metric']),
                            aggregate_df(df_f, ['demosaic', 'metric', 'filter', 'channels']),
                        ])
                        df_f = df_f[~df_f['channels'].isna()]
                        df_f['channels'] = df_f['channels'].astype(str)
                        df_f['model'] = model
                        df_f['inbayer'] = inbayer if inbayer else ''
                        df_f['information'] = df_f['filter'].apply(lambda f: 'Conditional' if inbayer else 'Unconditional')
                        # df_f['information'] = 'Conditional' if (len(df_f['filter'].split('_')) < 3) & bool(inbayer) else 'Unconditional'
                        df_f['information'] = df_f.apply(lambda r: r['information'] + (' with oracle' if has_oracle else ''), axis=1)
                        print(df_f)
                        df.append(df_f)

    df = pd.concat(df)
    df = (
        df
        .sort_values(['metric', 'demosaic', 'channels', 'model', 'filter'], axis=0)
        .reset_index(drop=True)
    )
    return df


def output_to_latex(df, results_dir):
    print(df)
    df['filter'] = df['filter'].apply(lambda f: f.split('_')[0])
    df['inbayer'] = df['inbayer'].fillna('')
    df = (
        aggregate_df(df, ['demosaic', 'metric', 'model', 'filter', 'channels', 'information'])
        .sort_values(['metric', 'demosaic', 'channels', 'model', 'filter'], axis=0)
        .reset_index(drop=True)
    )
    df['channel'] = df['channels'].apply(lambda s: s[0])
    df['channels'] = df['channels'].apply(lambda s: s.replace('0', 'R').replace('1', 'G').replace('2', 'B').replace('3', 'Y'))
    # df['#'] = df.apply(lambda r: (8 + (len(r['channels'])-1)*9) if r['filter'].startswith('OLS') else 0, axis=1)
    # df = df[df.metric == 'mae']
    df = df.pivot(columns=('information', 'demosaic'), index=('metric', 'filter', 'channels'), values=('value'))#, '#'), values=('value'))
    df = df.reindex(columns=['Unconditional', 'Conditional', 'Conditional with oracle'], level='information')
    df = df.reindex(columns=['*', 'linear', 'ahd', 'ppg', 'vng'], level='demosaic')

    df = df.reset_index(drop=False)
    df = df.sort_values(['metric', 'channels', 'filter'], ascending=True)
    df = df.fillna('')
    df['channels'] = df['channels'].apply(lambda c: r'\underline{'+c[0]+r'}'+c[1:])
    t = df.to_latex(
        # index=False,
        # header=False,
        float_format=lambda i: f'${i:.03f}$',
        index_names=False,
        escape=False,
    )
    print(df.to_string())
    print(t)
    # with open(results_dir / 'filter_prediction.tex', 'w') as f:
    #     f.write(t)


if __name__ == '__main__':
    #
    UNCONDITIONAL = ([None], False)
    CONDITIONAL = (['00', '01', '10', '11'], False)
    CONDITIONAL_ORACLE = (['00', '01', '10', '11'], True)
    take_num_images = None

    # BOSS
    results_dir = pathlib.Path('../results/filters_boss')
    input_dir = pathlib.Path('/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18/')
    models = ['gray']
    demosaics = [None]
    model_to_channels = {'gray': ((3,),)}
    model_to_filters = {
        'gray': [
            ['AVG']*4,
            ['KB']*4,
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
    print(f'OLS running on {os.cpu_count()} CPUs')

    df = evaluate_linear_filters(
        input_dir=input_dir,
        results_dir=results_dir,
        models=models,
        demosaics=demosaics,
        model_to_channels=model_to_channels,
        model_to_filters=model_to_filters,
        kernel_to_advantage=kernel_to_advantage,
        #
        take_num_images=take_num_images,
        # shuffle_seed=12345,
        split='split_te.csv',
        progress_on=False,
    )

    print(df.to_string())
    df.to_csv(results_dir / 'filter_prediction.csv', index=False)

    # filter prediction to latex
    df = pd.read_csv(results_dir / 'filter_prediction.csv', dtype={'channels': str, 'inbayer': str})
    output_to_latex(df, results_dir)