
import logging
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import typing

pd.options.mode.chained_assignment = None  # suppress Pandas warning


class CoverDataset(Dataset):
    """"""
    def __init__(
        self,
        # get dataset
        dataset_path: typing.Union[pathlib.Path, str],
        # get split
        config_name: str,
        # filter dataset
        filters_cover: typing.Dict[str, typing.Any],
        filters_cover_oneof: typing.Dict[str, typing.Any],
        filters_stego: typing.Dict[str, typing.Any],
        filters_stego_oneof: typing.Dict[str, typing.Any],
        # rotation
        rotation: int = None,
        # cover-stego distribution
        balance_batch: bool = True,
        pair_constraint: bool = False,
        seed: int = None,
        # additional
        imread: typing.Optional[typing.Callable] = lambda f: np.array(Image.open(f)),
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        drop_unpaired: bool = True,
        #
        debug: bool = False,  # returning filenames instead of images
    ):
        # parameters
        self.dataset_path = pathlib.Path(dataset_path)
        self.filters_cover = filters_cover
        self.filters_cover_oneof = filters_cover_oneof
        # self.filters_stego = filters_stego
        # self.filters_stego_oneof = filters_stego_oneof
        self.rotation = rotation
        # self.balance_batch = balance_batch
        # self.pair_constraint = pair_constraint
        self.imread = imread
        self.debug = debug
        self.transform = transform
        self.target_transform = target_transform
        self.drop_unpaired = drop_unpaired

        # get selected dataset
        self.config = pd.read_csv(self.dataset_path / config_name, low_memory=False)

        # filter based on parameters
        config_cover, config_stego = self.filter_cover_stego(self.config)

        # set cover and stego
        self.config_cover = config_cover.reset_index(drop=False)
        # self.config_stego = config_stego.reset_index(drop=False)

        # # drop unpaired samples
        # if self.drop_unpaired:
        #     self._sync_stego()

        #     no_cover = self.config_stego.index.difference(
        #         self.config_cover.index
        #     )
        #     no_stego = self.config_cover.index.difference(
        #         self.config_stego.index
        #     )
        #     for f in no_cover:
        #         logging.warning(f'dropping stego {f}, corresponding cover not found')
        #     for f in no_stego:
        #         logging.warning(f'dropping cover {f}, corresponding stego not found')
        #     self.config_cover = self.config_cover.drop(no_stego)
        #     self.config_stego = self.config_stego.drop(no_cover)

        #     self._reset_indices()

        # for dataset reshuffling
        self._rng = np.random.default_rng(seed)
        # self.reshuffle()

        # check dataset
        assert len(self.config_cover) > 0, 'no such covers found, did you forget running preprocess_dataset.py?'
        # assert len(self.config_stego) > 0, 'no such stegos found, did you forget running preprocess_dataset.py?'
        # assert len(self.config_cover) == len(self.config_stego), 'imbalanced cover-stego dataset'

    @staticmethod
    def basename(r):
        return f'{pathlib.Path(r["name"]).stem}_{int(r["rotation"])}'

    def __len__(self) -> int:
        """Dataset length."""
        # return len(self.config_cover) * 2
        return len(self.config_cover)

    def reshuffle(self):
        """Call after each epoch to reshuffle."""
        # shuffle cover
        seed = self._rng.integers(2**32-1)
        self.config_cover = self.config_cover.sample(
            frac=1,
            random_state=seed,
        )
        # # shuffle stego
        # if not self.pair_constraint:
        #     seed = self._rng.integers(2**32-1)
        # #     self.config_stego = self.config_stego.sample(
        # #         frac=1,
        # #         random_state=seed,
        # #     )
        # # sync stego
        # else:
        #     self._sync_stego()

    def filter_cover_stego(
        self,
        config: pd.DataFrame,
    ) -> typing.Tuple[pd.DataFrame]:
        """"""
        # filter covers
        for key, value in self.filters_cover.items():
            if key in config and value is not None:
                config = config[config[key] == value]
        for key, value in self.filters_cover_oneof.items():
            if key in config and value is not None:
                config = config[config[key].isin(value)]
        if 'alpha' in config:
            config['alpha'] = config['alpha'].fillna(0)
        # rotation
        if 'rotation' in config.columns:
            config[config.rotation.isnull()] = 0
            config['rotation'] = config['rotation'].astype(int)
        else:
            config['rotation'] = 0
        if self.rotation is None:
            config = config[config.rotation == 0]
        else:
            config = config[config.rotation == self.rotation]

        # separate cover and stego
        config_cover, config_stego = config[config.stego_method.isnull()], None

        # if 'stego_method' in config.columns:
        #     config_stego = config[~config.stego_method.isnull()]
        #     # filter stego
        #     for key, value in self.filters_stego.items():
        #         if value is not None:
        #             config_stego = config_stego[config_stego[key] == value]
        #     for key, value in self.filters_stego_oneof.items():
        #         if key in config_stego and value is not None:
        #             config_stego = config_stego[config_stego[key].isin(value)]

        # if self.balance_batch:
        #     config = (
        #         config_cover,
        #         config_stego
        #     )
        #     # config = (
        #     #     pd.concat([
        #     #         config_cover,
        #     #         config_stego,
        #     #     ]),
        #     #     None,
        #     # )
        # else:
        #     raise NotImplementedError
        #     config = (config_cover, config_stego)
        config = (config_cover, config_stego)
        assert len(config_cover) > 0, 'no such covers found'
        # assert config_stego is None or len(config_stego) > 0, 'no such stegos found'
        #
        return config

    def cover_index(self):
        df = self.config_cover.apply(self.basename, axis=1)
        return df

    # def stego_index(self):
    #     return self.config_stego.apply(self.basename, axis=1)

    def __contains__(self, name: str) -> bool:
        return name in self.config_cover.index

    def _reset_indices(self):
        """Reset indices of cover/stego datasets."""
        self.config_cover = self.config_cover.reset_index(drop=False)
        # self.config_stego = self.config_stego.reset_index(drop=False)

    def _sync_stego(self):
        """Order stego based on cover."""
        # convert to common key (basename)
        self.config_cover['fname'] = self.cover_index()
        self.config_cover = self.config_cover.set_index('fname')
        self.config_stego['fname'] = self.stego_index()
        self.config_stego = self.config_stego.set_index('fname')
        # sort jointly
        self.config_stego = self.config_stego.reindex(
            index=self.config_cover.index,
        )
        # # drop missing
        # self.config_stego = self.config_stego.dropna(subset=['name'])
        self._reset_indices()

    def __getitem__(self, idx: int):
        """Index dataset."""
        # cover or stego
        # is_stego = idx % 2 != 0  # zigzag cover-stego
        # config = self.config_stego if is_stego else self.config_cover
        # target = int(is_stego)
        config = self.config_cover
        # print(self.config_cover)

        # find cover
        image_row = config.iloc[idx, :]
        # idx_cover = np.argmax(self.config_cover['fname'] == image_row['fname'])
        # cover_row = self.config_cover.iloc[idx_cover, :]

        # load image
        cover_name = image_name = image_row['name']
        # cover_name = cover_row['name']
        image_alpha = image_row['alpha']
        if self.debug:
            return image_name, (cover_name, image_alpha)
        image = self.imread(self.dataset_path / image_name)# / 255.
        # cover = self.imread(self.dataset_path / cover_name)# / 255.
        cover = image.copy()
        # print(image[:3, :3])
        # print(image_name, cover_name, image_alpha)

        # transform
        if self.transform is not None:
            image = self.transform(image)
            # cover = self.transform(cover)
        if self.target_transform is not None:
            cover = self.target_transform(cover)
        target = (cover, image_alpha)

        # retreive image+target pairs
        return image, target

    def __iter__(self):
        """Iterate over cover-stego pairs."""
        for i in range(len(self) // 2):
            yield self[2*i]  # image
            yield self[2*i+1]  # cover


# for split in ['tr']:#, 'va', 'te']:

#     print(f'=== {split=} ===')
#     dataset = StegoDataset(
#         # dataset root
#         # '/Users/martin/UIBK/fabrika/alaska_20230923/',
#         '/gpfs/data/fs71999/uncover_mb/data/boss/fabrika-2024-01-18',
#         # config file
#         f'split_{split}.csv',
#         # training parameters
#         filters_cover={
#             'height': 512,
#             'width': 512,
#             # 'quality': None,
#         },
#         filters_cover_oneof={
#             # 'demosaic': ['ahd', 'ppg', 'vng'],
#         },
#         filters_stego={},
#         filters_stego_oneof={
#             'stego_method': ['LSBr'],
#             'alpha': [.2, .1],
#         },
#         seed=12345,  # for reshuffling
#         #
#         # demosaic_oracle=True,
#         debug=True,
#     )

#     for run in range(2):
#         print(f'- run {run+1} -')
#         dataset.reshuffle()
#         for i, (x, (y, alpha)) in enumerate(dataset):

#             if i > 20:
#                 break
#             print(x, y, alpha)
#             # print(x.shape, y.shape, alpha)
