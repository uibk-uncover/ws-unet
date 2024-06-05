
import glob
import hashlib
import joblib
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
import typing


class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, total: int = None, disable: bool = False, **kw):
        with tqdm(total=total, disable=disable) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kw)

    def print_progress(self):
        # self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def collect_files(
    patterns: typing.Sequence[str],
    fn: typing.Callable,
    pre_fn: typing.Callable = None,
    post_fn: typing.Callable = None,
    iterator: str = 'python',
    ignore_missing: bool = False,
    convert_to: bool = 'pandas',
    **kw_deco,
) -> typing.Callable:

    def iterate(
        dataset: pathlib.Path,
        skip_num_images: int = None,
        take_num_images: int = None,
        shuffle_seed: int = None,
        progress_on: bool = False,
        split: str = None,
        **kw_fn,
    ) -> pd.DataFrame:
        # get precovers
        dataset = pathlib.Path(dataset)

        if split is not None:
            df = pd.read_csv(dataset / split, dtype={'device': str})
        else:
            paths = []
            for pattern in patterns:
                paths += glob.glob(str(dataset / pattern))
            df = []
            for path in paths:
                try:
                    df.append(pd.read_csv(pathlib.Path(path) / 'files.csv'))
                except Exception:
                    if not ignore_missing:
                        raise
            df = pd.concat(df)

        # preprocess
        if pre_fn is not None:
            df = pre_fn(df, **kw_fn)
            if df.empty:
                raise Exception('pre_fn() returned empty dataframe')

        # take first
        df = df.sort_values('name').reset_index(drop=True)
        if shuffle_seed:
            df = df.sample(frac=1., random_state=shuffle_seed)
        if skip_num_images:
            df = df[skip_num_images:]
        if take_num_images:
            df = df[:take_num_images]

        # vanilla Python
        if iterator == 'python':
            res = []
            keys = df.columns.difference(pd.Index(['name']))
            for index, row in tqdm(df.iterrows(), total=len(df), disable=not progress_on):
                res.append(fn(
                    dataset / row['name'],
                    **(row.to_dict() | kw_fn)
                ))

        # joblib
        elif iterator == 'joblib':
            gen = (
                joblib.delayed(fn)(
                    dataset / row['name'],
                    **(row.to_dict() | kw_fn),
                )
                for index, row in df.iterrows()
            )
            res = ProgressParallel(**kw_deco)(gen, total=len(df), disable=not progress_on)

        # provide entire dataframe
        elif iterator is None:
            if progress_on:
                tqdm.tqdm.pandas()
                df['name'] = df['name'].progress_apply(lambda d: str(dataset / d))
            else:
                df['name'] = df['name'].apply(lambda d: str(dataset / d))
            res = fn(df, **kw_fn)

        else:
            raise NotImplementedError(f'unknown iterator {iterator}')

        # convert
        if convert_to is None:
            pass
        elif convert_to == 'pandas':
            res = pd.DataFrame(res)
        elif convert_to == 'numpy':
            res = np.array(res)
        else:
            raise NotImplementedError(f'unknown convertor {convert_to}')

        # postprocess
        if post_fn is not None:
            res = post_fn(res, **kw_fn)

        return res

    return iterate


def precovers(**kw_deco):
    def _precovers(fn: typing.Callable):
        def pre_fn(
            df,
            demosaic: str = None,
            *args,
            **kw,
        ) -> pd.DataFrame:
            if demosaic is None:
                pass
            elif isinstance(demosaic, str):
                df = df[df['demosaic'] == demosaic]
            else:
                df = df[df['demosaic'].isin(demosaic)]
            if 'stego_method' in df:
                df = df[df['stego_method'].isna()]
            if 'quality' in df:
                df = df[df['quality'].isna()]
            return df

        return collect_files(['images*'], fn=fn, pre_fn=pre_fn, **kw_deco)
    return _precovers


def covers(**kw_deco):
    def _covers(fn: typing.Callable):
        def pre_fn(
            df: pd.DataFrame,
            quality: int = None,
            samp_factor: str = None,
            **kw,
        ) -> pd.DataFrame:
            if quality is not None:
                df = df[df['quality'] == f'q{quality}']
            if samp_factor is not None:
                df = df[df['samp_factor'] == samp_factor]
            return df

        return collect_files(['jpegs*'], fn=fn, pre_fn=pre_fn, **kw_deco)
    return _covers


def stego_spatial(**kw_deco):
    def _stego_spatial(fn: typing.Callable):
        def pre_fn(
            df: pd.DataFrame,
            stego_method: int = None,
            alpha: str = None,
            color_strategy: str = None,
            simulator: str = None,
            demosaic: str = None,
            **kw,
        ) -> pd.DataFrame:
            if demosaic is None:
                pass
            elif isinstance(demosaic, str):
                df = df[df['demosaic'] == demosaic]
            else:
                df = df[df['demosaic'].isin(demosaic)]
            if stego_method is not None:
                # print(stego_method)
                # print(df)
                df = df[df['stego_method'] == stego_method]
            if alpha is not None:
                df = df[df['alpha'] == alpha]
            if color_strategy is not None:
                df = df[df['color_strategy'] == color_strategy]
            if simulator is not None:
                df = df[df['simulator'] == simulator]
            if 'quality' in df:
                df = df[df['quality'].isna()]
            return df

        return collect_files(['stego*'], fn=fn, pre_fn=pre_fn, **kw_deco)
    return _stego_spatial


def cover_stego_spatial(paired=True, **kw_deco):
    def _cover_stego_spatial(fn: typing.Callable):
        def pre_fn(
            df: pd.DataFrame,
            stego_method: int = None,
            alpha: str = None,
            color_strategy: str = None,
            simulator: str = None,
            demosaic: str = None,
            **kw,
        ) -> pd.DataFrame:
            # filter cover types
            if demosaic is None:
                pass
            elif isinstance(demosaic, str):
                df = df[df['demosaic'] == demosaic]
            else:
                df = df[df['demosaic'].isin(demosaic)]
            if 'quality' in df:
                df = df[df['quality'].isna()]

            # split cover and stegos
            df_c = df[df['stego_method'].isna()]
            df_s = df[~df['stego_method'].isna()]

            # filter stego
            if stego_method is not None:
                df_s = df_s[df_s['stego_method'] == stego_method]
            if alpha is not None:
                df_s = df_s[df_s['alpha'] == alpha]
            if color_strategy is not None:
                df_s = df_s[df_s['color_strategy'] == color_strategy]
            if simulator is not None:
                df_s = df_s[df_s['simulator'] == simulator]

            # paired
            if paired:
                df_c['stem'] = df_c['name'].apply(lambda f: pathlib.Path(f).stem)
                df_s['stem'] = df_s['name'].apply(lambda f: pathlib.Path(f).stem)
                df = df_c.merge(df_s, how='left', on=['stem'], suffixes=('_c', '_s'))
                df = df.drop('stem', axis=1)
                df['name'] = df['name_c']

            else:
                raise NotImplementedError

            return df

        def post_fn(
            df: pd.DataFrame,
            *args,
            **kw,
        ) -> pd.DataFrame:
            df['stem'] = df['name_c'].apply(lambda f: pathlib.Path(f).stem)
            df = df.sort_values(['stem', 'name_c'])
            df = df.drop('stem', axis=1)
            return df

        return collect_files(
            ['images*', 'stego*'],
            fn=fn,
            pre_fn=pre_fn,
            post_fn=post_fn,
            **kw_deco,
        )
    return _cover_stego_spatial


def filename_to_image_seed(filename: str):
    """Generate component seeds from the filename stem."""

    # SHA256 of file stem (basename)
    filename_stem = pathlib.Path(filename).stem

    # Encode as bytes
    filename_stem_bytes = filename_stem.encode('utf-8')

    # Hash stem
    sha256 = hashlib.sha256(filename_stem_bytes).hexdigest()

    # Hash to seed (0-2^31, if seed is signed)
    image_seed = int(sha256, base=16) % (2**31)

    return image_seed
