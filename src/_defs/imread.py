
import cv2
import jpeglib
from PIL import Image
import numpy as np


def imread_u8(fname) -> np.ndarray:
    x = np.array(Image.open(fname))
    if len(x.shape) == 2:
        x = x[..., None]
    return x


def imread_f32(fname) -> np.ndarray:
    return imread_u8(fname).astype('float32')


def imread4_u8(fname: str) -> np.ndarray:
    x_bgr = cv2.imread(str(fname))
    x_y = cv2.cvtColor(x_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    x = np.concatenate([x_bgr[..., ::-1], x_y], axis=-1)
    return x


def imread4_f32(fname: str) -> np.ndarray:
    return imread4_u8(fname).astype('float32')


# def imread_pillow(f):
#     x = np.array(Image.open(f))
#     if len(x.shape) == 2:
#         x = x[..., None]
#     return x


# def imread_jpeglib(f):
#     return jpeglib.read_spatial(f).spatial


# def imread_jpeglib_YCbCr(f):
#     return jpeglib.read_spatial(f, jpeglib.JCS_YCbCr).spatial
