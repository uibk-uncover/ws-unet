
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import typing


class RandomRotation90(torch.nn.Module):
    """Rotate the image by a random multiple of 90, clock-wise."""

    def __init__(self, p=torch.tensor([.25, .25, .25, .25])):
        self.p = p / torch.sum(p)
        self.cump = torch.cumsum(self.p, 0)

    def get_params(self) -> float:
        """Get parameters for ``rotate`` for a random rotation."""
        x = torch.empty(1).uniform_()
        i = int(torch.argmax((x < self.cump).to(torch.uint8)))
        return i

    def __call__(self, img):
        i = self.get_params()
        return F.rotate(img, i * 90, F.InterpolationMode.NEAREST)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(p={self.p}"
        format_string += ")"
        return format_string


class ColorChannel(torch.nn.Module):
    """Select color channel(s)."""

    def __init__(self, channels):
        if isinstance(channels, int):
            self.channels = [channels]
        else:
            self.channels = channels

    def __call__(self, img):
        return img[self.channels]

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(channels={self.channels})"


class Grayscale(transforms.Grayscale):
    def forward(self, img):
        if img.shape[0] == 1:
            return img
        elif img.shape[0] == 4:
            return img[3:]
        else:
            return super().forward(img)


class LSBrReference(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        lsbr_reference = (torch.round(img * 255).int() & ~1)/255.
        return torch.cat([img, lsbr_reference], dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class ParityOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        demosaic_grid = torch.round(img * 255).int() & 1
        return torch.cat([img, demosaic_grid], dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class DemosaicOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        demosaic_grid = torch.zeros(3, *img.shape[1:])
        demosaic_grid[0, ::2, ::2] = 1
        demosaic_grid[1, 1::2, ::2] = 1
        demosaic_grid[1, ::2, 1::2] = 1
        demosaic_grid[2, 1::2, 1::2] = 1
        # demosaic_grid = torch.zeros(4, *img.shape[1:])
        # demosaic_grid[0, ::2, ::2] = 1
        # demosaic_grid[1, 1::2, ::2] = 1
        # demosaic_grid[2, ::2, 1::2] = 1
        # demosaic_grid[3, 1::2, 1::2] = 1
        return torch.cat([img, demosaic_grid], dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RandomRotationDataset(Dataset):  # (ChainDataset):
    def __init__(
        self,
        datasets: typing.Sequence[Dataset],
        augment_seed: int = None,
    ):
        # super().__init__(datasets)
        self.datasets = datasets
        self.D = len(self.datasets)

        # get indices
        indices = [d.cover_index() for d in self.datasets]

        # merge non-duplicated indices, sort
        self.index = (
            pd.concat(indices)
            .drop_duplicates()
            .sort_values()
        )

        # remove missing
        missing = []
        for c in self.index:
            if any([c not in d for d in self.datasets]):
                missing.append(c)
        for f in missing:
            logging.warning(f'dropping {f}, some rotations missing')
        self.index = self.index.drop(missing)

        # for rotation selection
        self._rng = np.random.default_rng(augment_seed)
        self._perm = list(range(len(self)))
        self.rotations = np.zeros(len(self), dtype='uint8')

    def reshuffle(self):
        """"""
        # shuffle
        self._rng.shuffle(self._perm)
        self.rotations = self._rng.choice(
            range(self.D),
            size=len(self._perm),
        )

        # shuffle datasets
        for d in self.datasets:
            d.reshuffle()

    def __getitem__(self, idx: int):

        # get permuted dataset index
        perm_idx = self._perm[idx]

        # index rotation
        d_idx = self.rotations[perm_idx]

        # get dataset
        return self.datasets[d_idx][perm_idx]

    # def __iter__(self, *args, **kw):
    #     """"""
    #     # generate random rotations
    #     d_indices = self._rng.choice(range(self.D), size=len(self._perm))

    #     # iterate covers
    #     print(f'args/kw:', args, kw, len(self._perm))
    #     for idx in self._perm:

    #         # index rotation
    #         d_idx = d_indices[idx]

    #         # get dataset
    #         yield self.datasets[d_idx][2*idx]
    #         yield self.datasets[d_idx][2*idx+1]

    def __len__(self) -> int:
        return 2 * len(self.index)
