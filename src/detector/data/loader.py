
import jpeglib
import json
import logging
import numpy as np
import pandas as pd
import pathlib
import sys
import timm
import torch
from torch.utils.data import Dataset, ChainDataset
import torchvision.transforms as transforms
import typing

sys.path.append('..')
import _defs

# try:
#     from . import StegoDataset
# except ImportError:
#     # sys.path.append('.')
#     # sys.path.append('detector')
#     # sys.path.append('detector/data')
#     # import StegoDataset
# sys.path.append('..')
# import _defs

# sys.path.append('..')
# from _defs import RandomRotation90, ColorChannel, Grayscale, DemosaicOracle
# from _defs import imread_pillow, imread_jpeglib_YCbCr, imread_jpeglib


def get_timm_transform(
    mean: float,
    std: float,
    grayscale: bool = False,
    demosaic_oracle: bool = False,
    post_flip: bool = False,
    post_rotate: bool = False,
    lsbr_reference: bool = False,
):
    # dataset transforms
    transform = [
        transforms.ToTensor(),  # to torch tensor
        transforms.CenterCrop(512),  # reduce large images to 512x512
    ]
    # convert to grayscale
    if grayscale:
        transform.append(_defs.Grayscale())
    if lsbr_reference:
        transform.append(_defs.LSBrReference())
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


# def get_data_loader(
#     config_path: pathlib.Path,
#     args: typing.Dict[str, typing.Any],
#     augment: bool = False,
#     debug: bool = False,
# ):
#     """"""
#     # Normalization
#     mean = list(timm.data.constants.IMAGENET_DEFAULT_MEAN)
#     std = list(timm.data.constants.IMAGENET_DEFAULT_STD)
#     if args['grayscale']:  # take green
#         mean = mean[1:2]
#         std = std[1:2]

#     # Reader
#     if args['quality'] is None:
#         imread = _defs.imread4_u8
#     else:
#         raise NotImplementedError
#     #     imread = imread_pillow
#     # elif not args['grayscale']:
#     #     imread = imread_jpeglib_YCbCr
#     # else:
#     #     imread = imread_jpeglib

#     # Dataset transform
#     transform = get_timm_transform(
#         mean=mean,
#         std=std,
#         grayscale=args.get('grayscale', False),
#         demosaic_oracle=args.get('demosaic_oracle', False),
#         post_flip=args.get('post_flip', False),
#         post_rotate=args.get('post_rotate', False),
#         lsbr_reference=args.get('lsbr_reference', False),
#     )
#     target_transform = None
#     print('Data transform')
#     print(transform)

#     # Dataset
#     dataset = StegoDataset(
#         # dataset
#         args['dataset'],
#         config_path,
#         imread=imread,
#         # training parameters
#         filters_cover=args['filters_cover'],
#         filters_cover_oneof=args['filters_cover_oneof'],
#         filters_stego=args['filters_stego'],
#         filters_stego_oneof=args['filters_stego_oneof'],
#         rotation=None if args['pre_rotate'] else 0,  #
#         # pair constraint
#         pair_constraint=args['pair_constraint'],
#         seed=args['seed'],  # for shuffling, if PC=False
#         # other
#         transform=transform,
#         target_transform=target_transform,
#         debug=debug,
#     )

#     # Create data loaders
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args["batch_size"],
#         shuffle=False,
#         num_workers=args["num_workers"],
#         pin_memory=True,
#     )

#     #
#     return loader, dataset


# for split in ['tr', 'va', 'te']:
#     split = 'te'
#     print(f'{split=}')
#     datasets = []
#     for rotate in range(4):
#         dataset = StegoDataset(
#             # dataset root
#             '/gpfs/data/fs71999/uncover_bl/data/alaska2/fabrika-2023-04-14',
#             # config file
#             f'config/bl/split_{split}.csv',
#             # training parameters
#             shape=(512, 512),
#             quality=75,
#             stego_method='J-UNIWARD',
#             alpha=.4, beta=.0,
#             rotation=rotate * 90,
#             # pair constraint
#             pair_constraint=False,
#             seed=12345,  # for shuffling, if PC=False
#             #
#             debug=True,
#         )

#         datasets.append(dataset)

#     # create rotation cover-stego loader
#     rot_dataset = RotationDataset(datasets)
#     loader = torch.utils.data.DataLoader(
#         rot_dataset,
#         batch_size=8,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#     )

#     rot_dataset.reshuffle()
#     for i, f in enumerate(loader):
#         print(f)
#         break


#     # print('run 2')
#     # dataset.reshuffle()
#     # for i, x in enumerate(dataset):
#     #     if i > 6:
#     #         break
#     #     print(x)

#     break
