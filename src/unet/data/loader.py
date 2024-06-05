
import jpeglib
import json
import logging
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import sys
import timm
import torch
from torch.utils.data import Dataset, ChainDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import typing

sys.path.append('..')
import _defs

try:
    from .stego_dataset import StegoDataset
    from .cover_dataset import CoverDataset
except ImportError:
    from stego_dataset import StegoDataset
    from cover_dataset import CoverDataset


# from _defs import RandomRotation90, ColorChannel, Grayscale, DemosaicOracle
# from _defs import imread_pillow, imread_jpeglib_YCbCr, imread_jpeglib


def get_timm_transform(
    mean: float,
    std: float,
    grayscale: bool = False,
    parity_oracle: bool = False,
    demosaic_oracle: bool = False,
    post_flip: bool = False,
    post_rotate: bool = False,
):
    # datast transforms
    transform = [
        transforms.ToTensor(),  # to torch tensor
        transforms.CenterCrop(512),  # reduce large images to 512x512
    ]
    # convert to grayscale
    if grayscale:
        transform.append(_defs.Grayscale())
    if parity_oracle:
        transform.append(_defs.ParityOracle())
    if demosaic_oracle:
        transform.append(_defs.DemosaicOracle())
    # normalize by given moments
    if mean is not None and std is not None:
        transform.append(transforms.Normalize(mean, std))
    # augment
    if post_flip:
        transform += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    if post_rotate:
        transform += [_defs.RandomRotation90()]
    return transforms.Compose(transform)


# class RotationDataset(Dataset):
#     def __init__(
#         self,
#         datasets: typing.Sequence[Dataset],
#         augment_seed: int = None,
#     ):
#         # super().__init__(datasets)
#         self.datasets = datasets
#         self.D = len(self.datasets)

#         # get indices
#         indices = [d.cover_index() for d in self.datasets]

#         # merge non-duplicated indices, sort
#         self.index = (
#             pd.concat(indices)
#             .drop_duplicates()
#             .sort_values()
#         )

#         # remove missing
#         missing = []
#         for c in self.index:
#             if any([c not in d for d in self.datasets]):
#                 missing.append(c)
#         for f in missing:
#             logging.warning(f'dropping {f}, some rotations missing')
#         self.index = self.index.drop(missing)

#         # for rotation selection
#         self._rng = np.random.default_rng(augment_seed)
#         self._perm = list(range(len(self)))
#         self.rotations = np.zeros(len(self), dtype='uint8')

#     def reshuffle(self):
#         """"""
#         # shuffle
#         self._rng.shuffle(self._perm)
#         self.rotations = self._rng.choice(
#             range(self.D),
#             size=len(self._perm),
#         )

#         # shuffle datasets
#         for d in self.datasets:
#             d.reshuffle()

#     def __getitem__(self, idx: int):

#         # get permuted dataset index
#         perm_idx = self._perm[idx]

#         # index rotation
#         d_idx = self.rotations[perm_idx]

#         # get dataset
#         return self.datasets[d_idx][perm_idx]

#     def __len__(self) -> int:
#         return 2 * len(self.index)


def get_data_loader(
    config_path: pathlib.Path,
    args: typing.Dict[str, typing.Any],
    augment: bool = False,
    debug: bool = False,
):
    """"""
    # Normalization
    mean = list(timm.data.constants.IMAGENET_DEFAULT_MEAN)
    std = list(timm.data.constants.IMAGENET_DEFAULT_STD)
    # mean, std = [0, 0, 0], [1, 1, 1]
    mean, std = None, None
    # if args['grayscale']:  # take green
    #     mean = mean[1:2]
    #     std = std[1:2]

    # Reader
    imread = _defs.imread4_u8
    # imread = _defs.imread_u8
    # if args['quality'] is None:
    #     imread = imread_pillow
    # elif not args['grayscale']:
    #     imread = imread_jpeglib_YCbCr
    # else:
    #     imread = imread_jpeglib

    # Dataset transform
    transform = get_timm_transform(
        mean=mean,
        std=std,
        grayscale=args.get('grayscale', False),
        demosaic_oracle=args.get('demosaic_oracle', False),
        parity_oracle=args.get('parity_oracle', False),
        post_flip=args.get('post_flip', False),
        post_rotate=args.get('post_rotate', False),
    )
    target_transform = transforms.Compose([
        transforms.ToTensor(),  # to torch tensor
        transforms.CenterCrop(512),  # reduce large images to 512x512
        _defs.ColorChannel(args['channel']),
        # transforms.Normalize(mean, std),
    ])
    print('Data transform')
    print(transform)
    print(target_transform)

    # Dataset
    if args.get('covers_only', False):
        Dataset = CoverDataset
    else:
        Dataset = StegoDataset

    dataset = Dataset(
        # dataset
        args['dataset'],
        config_path,
        imread=imread,
        # training parameters
        filters_cover=args['filters_cover'],
        filters_cover_oneof=args['filters_cover_oneof'],
        filters_stego=args['filters_stego'],
        filters_stego_oneof=args['filters_stego_oneof'],
        rotation=None if args['pre_rotate'] else 0,  #
        # pair constraint
        pair_constraint=args['pair_constraint'],
        seed=args['seed'],  # for reshuffling
        # other
        transform=transform,
        target_transform=target_transform,
        debug=debug,
    )

    # Data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    #
    return loader, dataset


# for split in ['tr', 'va', 'te']:
#     split = 'te'
#     print(f'{split=}')
#     # datasets = []
#     # for rotate in range(1):
#     #     dataset = StegoDataset(
#     #         # dataset root
#     #         '/Users/martin/UIBK/fabrika/alaska_20230923/',
#     #         # '/gpfs/data/fs71999/uncover_bl/data/alaska2/fabrika-2023-04-14',
#     #         # config file
#     #         # f'config/bl/split_{split}.csv',
#     #         f'split_{split}.csv',
#     #         # training parameters
#     #         filters_cover={
#     #             'height': 512,
#     #             'width': 512,
#     #             'demosaic': 'linear'
#     #         },
#     #         filters_stego={
#     #             'stego_method': 'LSBr',
#     #             'alpha': .4,
#     #         },
#     #         rotation=rotate * 90,
#     #         # pair constraint
#     #         pair_constraint=False,
#     #         seed=12345,  # for shuffling, if PC=False
#     #         #
#     #         debug=True,
#     #     )

#     #     datasets.append(dataset)

#     # # create rotation cover-stego loader
#     # dataset = RotationDataset(datasets)

#     transform = get_timm_transform(
#         3,
#         mean=None,
#         std=None,
#         post_flip=False,
#         post_rotate=False,
#     )
#     target_transform = transforms.Compose([
#         ColorChannel(0)
#     ])
#     dataset = StegoDataset(
#         # dataset root
#         # '/Users/martin/UIBK/fabrika/alaska_20230923/',
#         '/gpfs/data/fs71999/uncover_mb/data/alaska2/fabrika-2023-12-18',
#         # config file
#         # f'config/bl/split_{split}.csv',
#         f'split_{split}.csv',
#         # training parameters
#         filters_cover={
#             'height': 512,
#             'width': 512,
#         },
#         filters_cover_oneof={
#             'demosaic': ['ahd', 'ppg', 'vng']
#         },
#         filters_stego={
#             'stego_method': 'LSBr',
#             'alpha': .4,
#         },
#         rotation=0,
#         # pair constraint
#         pair_constraint=False,
#         seed=12345,  # for shuffling, if PC=False
#         #
#         transform=transform,
#         target_transform=target_transform,
#         demosaic_oracle=True,
#         # debug=True,
#     )

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#     )

#     dataset.reshuffle()
#     for i, (x, y) in enumerate(loader):
#         print(x.shape, y.shape)
#         print(x[:, :, 0, 0])
#         break

#     break
