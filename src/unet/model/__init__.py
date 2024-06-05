
import pathlib
import torch

from . import unet


def get_model(
    name: str,
    in_channels: int,
    out_channels: int = 1,
    channel: int = [0],
    drop_rate: float = 0.,
    # dropout_type: str = 'uniform',
) -> torch.nn.Module:

    # U-Net
    if name.lower().startswith('unet'):
        nsteps = int(name.split('_')[1])
        model = unet.UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            nsteps=nsteps,
            drop_channel=channel,
            # dropout_type=dropout_type,
            drop_rate=drop_rate,
        )

    # Toy CNN
    elif name.lower().startswith('cnn'):
        model = unet.CNN(
            in_channels=in_channels,
            drop_channel=channel,
            drop_rate=drop_rate,
        )

        kb = torch.FloatTensor([[[
            [-1, +2, -1],
            [+2,  0, +2],
            [-1, +2, -1],
        ]]]) / 4.
        model.e11.weight = torch.nn.Parameter(kb)
        # model.e11.weight.requires_grad = False

    # unknown network
    else:
        raise NotImplementedError(name)

    return model


def load_model(
    model_path: pathlib.Path,
    model_name: str,
    device: torch.nn.Module,
    **kw
) -> torch.nn.Module:
    model = get_model('unet_1', **kw).to(device)
    resume_model_file = model_path / model_name / 'model' / 'best_model.pt.tar'
    checkpoint = torch.load(resume_model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
