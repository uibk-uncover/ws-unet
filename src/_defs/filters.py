
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import typing


THETAS_PER_MODEL = {
    'gray': ('rho',),
    'color4': ('rho', 'chi'),
    'color8': ('rho', 'chi'),
}

BETAS_PER_MODEL = {
    'gray': ('x00', 'x01', 'x02', 'x12', 'x22', 'x21', 'x20', 'x10', 'x11'),
    'color4': (
        'y00', 'y01', 'y02', 'y12', 'y22', 'y21', 'y20', 'y10', 'y11',
        'x00', 'x01', 'x02', 'x12', 'x22', 'x21', 'x20', 'x10', 'x11',
    ),
    'color8': (
        'z00', 'z01', 'z02', 'z12', 'z22', 'z21', 'z20', 'z10', 'z11',
        'y00', 'y01', 'y02', 'y12', 'y22', 'y21', 'y20', 'y10', 'y11',
        'x00', 'x01', 'x02', 'x12', 'x22', 'x21', 'x20', 'x10', 'x11',
    ),
}

DENSITY_VARIABLES = {
    'gray': ('beta_x00', 'beta_x01'),
    'color4': ('beta_y00', 'beta_y01', 'beta_y11', 'beta_x00', 'beta_x01'),
    'color8': ('beta_z00', 'beta_z01', 'beta_z11', 'beta_y00', 'beta_y01', 'beta_y11', 'beta_x00', 'beta_x01'),
}


INBAYERS = ['00', '01', '10', '11']


def get_processor(
    channels: typing.List[int],
    inbayer: str = None,
) -> typing.Callable:
    # parse parameters
    step = 1
    bayer1_start, bayer1_end, bayer2_start, bayer2_end = None, None, None, None
    if inbayer:
        step = 2
        if inbayer[0] == '0':
            bayer1_start, bayer1_end = 1, -1
        if inbayer[1] == '0':
            bayer2_start, bayer2_end = 1, -1

    def process_gray(x: np.ndarray) -> np.ndarray:
        x = x[bayer1_start:bayer1_end, bayer2_start:bayer2_end]
        # print(f'process_gray {inbayer} crop [{bayer1_start}:{bayer1_end},{bayer2_start}:{bayer2_end}] -> {x.shape}')
        # print(channels)
        return np.stack([
            x[:-2:step, :-2:step, channels[0]].flatten(),  # x00
            x[:-2:step, 1:-1:step, channels[0]].flatten(),  # x01
            x[:-2:step, 2::step, channels[0]].flatten(),  # x02
            x[1:-1:step, 2::step, channels[0]].flatten(),  # x12
            x[2::step, 2::step, channels[0]].flatten(),  # x22
            x[2::step, 1:-1:step, channels[0]].flatten(),  # x21
            x[2::step, :-2:step, channels[0]].flatten(),  # x20
            x[1:-1:step, :-2:step, channels[0]].flatten(),  # x10
            x[1:-1:step, 1:-1:step, channels[0]].flatten(),  # x11
        ], axis=-1)

    return process_gray


def get_processor_2d(
    channels: typing.List[int],
) -> typing.Callable:
    # parse parameters
    step = 1
    bayer1_start, bayer1_end, bayer2_start, bayer2_end = None, None, None, None

    def process_gray(x: np.ndarray) -> np.ndarray:
        x = x[bayer1_start:bayer1_end, bayer2_start:bayer2_end]
        return x[::step, ::step, channels].astype('float32')

    return process_gray
