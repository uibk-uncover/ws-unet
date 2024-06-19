"""Implementation of UNet architectre.

This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

Taken from
https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
"""

import torch
import torch.nn.functional as F
from torch import nn


class UniformDropout(torch.nn.Module):
    """Custom dropout."""
    def __init__(self, p, drop_channel):
        super().__init__()
        # print(f'UniformDropout({p=})')
        # self.p = p
        # self.p = torch.nn.Parameter(torch.tensor(p), requires_grad=False)
        self.p = 1-p
        self.kb = torch.tensor([[[
            [-1, +2, -1],
            [+2, +0, +2],
            [-1, +2, -1]
        ]]], dtype=torch.float32) / 4.
        self.drop_channel = drop_channel
        self.mask = None
        self.device = None

    def forward(self, x):
        c = self.drop_channel
        self.mask = torch.empty(x.shape[0], 1, *x.shape[2:]).bernoulli_(p=self.p)
        self.mask = self.mask.repeat((1, len(c), 1, 1))
        if self.device:
            self.mask = self.mask.to(self.device)
        #
        x_pad = F.pad(x[:, c], (1, 1, 1, 1), mode='reflect')
        x_kb = F.conv2d(x_pad, self.kb.repeat((1, len(c), 1, 1)))
        x[:, c] = x[:, c] * self.mask + x_kb * (1 - self.mask)
        return x
        # #
        # x[:, c] = x[:, c] * self.mask
        # return x / self.p

    def to(self, device, *args, **kw):
        super().to(*args, device, **kw)
        self.device = device
        self.kb = self.kb.to(self.device)
        return self


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsteps: int,
        drop_rate: float,
        # dropout_type: str,
        drop_channel: int,
    ):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------

        conv_kw = {'kernel_size': 3, 'padding': 1, 'padding_mode': 'reflect'}
        ups_kw = {'kernel_size': 2, 'stride': 2}  # , 'padding_mode': 'reflect'}  # only zero-padding possible
        Dropout = UniformDropout

        # input: 572x572x3
        if drop_rate is not None:
            self.input_dropout = Dropout(p=drop_rate, drop_channel=drop_channel)  # randomly drop input pixels
        else:
            self.input_dropout = None
        self.e11 = nn.Conv2d(in_channels, 64, **conv_kw)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, **conv_kw)  # output: 568x568x64

        if self.nsteps >= 1:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

            # input: 284x284x64
            self.e21 = nn.Conv2d(64, 128, **conv_kw)  # output: 282x282x128
            self.e22 = nn.Conv2d(128, 128, **conv_kw)  # output: 280x280x128

        if self.nsteps >= 2:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

            # input: 140x140x128
            self.e31 = nn.Conv2d(128, 256, **conv_kw)  # output: 138x138x256
            self.e32 = nn.Conv2d(256, 256, **conv_kw)  # output: 136x136x256

        if self.nsteps >= 3:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

            # input: 68x68x256
            self.e41 = nn.Conv2d(256, 512, **conv_kw)  # output: 66x66x512
            self.e42 = nn.Conv2d(512, 512, **conv_kw)  # output: 64x64x512

        if self.nsteps >= 4:
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

            # input: 32x32x512
            self.e51 = nn.Conv2d(512, 1024, **conv_kw)  # output: 30x30x1024
            self.e52 = nn.Conv2d(1024, 1024, **conv_kw)  # output: 28x28x1024

        # Decoder
        if self.nsteps >= 4:
            self.upconv1 = nn.ConvTranspose2d(1024, 512, **ups_kw)
            self.d11 = nn.Conv2d(1024, 512, **conv_kw)
            self.d12 = nn.Conv2d(512, 512, **conv_kw)

        if self.nsteps >= 3:
            self.upconv2 = nn.ConvTranspose2d(512, 256, **ups_kw)
            self.d21 = nn.Conv2d(512, 256, **conv_kw)
            self.d22 = nn.Conv2d(256, 256, **conv_kw)

        if self.nsteps >= 2:
            self.upconv3 = nn.ConvTranspose2d(256, 128, **ups_kw)
            self.d31 = nn.Conv2d(256, 128, **conv_kw)
            self.d32 = nn.Conv2d(128, 128, **conv_kw)

        if self.nsteps >= 1:
            self.upconv4 = nn.ConvTranspose2d(128, 64, **ups_kw)  # zero padding!
            self.d41 = nn.Conv2d(128, 64, **conv_kw)
            self.d42 = nn.Conv2d(64, 64, **conv_kw)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1, padding_mode='reflect')

    def forward(self, x_in):
        # Encoder
        if self.input_dropout is not None:
            x_in = self.input_dropout(x_in)
        xe11 = F.relu(self.e11(x_in))
        x = xe12 = F.relu(self.e12(xe11))
        if self.nsteps >= 1:
            xp1 = self.pool1(xe12)
            xe21 = F.relu(self.e21(xp1))
            x = xe22 = F.relu(self.e22(xe21))

        if self.nsteps >= 2:
            xp2 = self.pool2(xe22)
            xe31 = F.relu(self.e31(xp2))
            x = xe32 = F.relu(self.e32(xe31))

        if self.nsteps >= 3:
            xp3 = self.pool3(xe32)
            xe41 = F.relu(self.e41(xp3))
            x = xe42 = F.relu(self.e42(xe41))

        if self.nsteps >= 4:
            xp4 = self.pool4(xe42)
            xe51 = F.relu(self.e51(xp4))
            x = xe52 = F.relu(self.e52(xe51))

        # Decoder
        if self.nsteps >= 4:
            xu1 = self.upconv1(x)
            xu11 = torch.cat([xu1, xe42], dim=1)
            xd11 = F.relu(self.d11(xu11))
            x = xd12 = F.relu(self.d12(xd11))

        if self.nsteps >= 3:
            xu2 = self.upconv2(x)
            xu22 = torch.cat([xu2, xe32], dim=1)
            xd21 = F.relu(self.d21(xu22))
            x = xd22 = F.relu(self.d22(xd21))

        if self.nsteps >= 2:
            xu3 = self.upconv3(x)
            xu33 = torch.cat([xu3, xe22], dim=1)
            xd31 = F.relu(self.d31(xu33))
            x = xd32 = F.relu(self.d32(xd31))

        if self.nsteps >= 1:
            xu4 = self.upconv4(x)
            xu44 = torch.cat([xu4, xe12], dim=1)
            xd41 = F.relu(self.d41(xu44))
            x = F.relu(self.d42(xd41))

        # Output layer
        return F.sigmoid(self.outconv(x))

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.input_dropout.to(*args, **kw)
        return self

    def disable_center_pixels(self):
        self.e11.weight.data[:, :, 1, 1] = 0.
        if self.e11.weight.grad is not None:
            self.e11.weight.grad[:, :, 1, 1] = 0.
