"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import pathlib
import timm
import torch
from typing import Tuple


def get_b0(
    # name: str,
    # weights: str,
    in_channels: int,
    shape: Tuple[int],
    num_classes: int = 2,
    pretrained: bool = False,
    strict: bool = False,
    no_stem_stride: bool = False,
    **kw
) -> torch.nn.Module:

    # EfficientNetB0
    model = timm.create_model(
        'efficientnet_b0',
        num_classes=num_classes,
        in_chans=in_channels,
        pretrained=pretrained,
        **kw,
    )
    model.model_name = 'b0'
    fc_name = 'classifier'
    conv_stem_names = ['conv_stem']

    if no_stem_stride:
        model.conv_stem.stride = (1, 1)

    # load weights
    state_dict = torch.hub.load_state_dict_from_url(model.default_cfg['url'])

    # remove FC, if not compatible
    out_fc, _ = state_dict[fc_name + '.weight'].shape
    if out_fc != num_classes:
        del state_dict[fc_name + '.weight']
        del state_dict[fc_name + '.bias']

    # modify first convolution to match the input size
    for conv_stem_name in conv_stem_names:
        weight_name = conv_stem_name + '.weight'
        _, in_conv, _, _ = state_dict[weight_name].shape
        if in_conv != in_channels:
            state_dict[weight_name] = timm.models.adapt_input_conv(in_channels, state_dict[weight_name])

    state_dict['input_size'] = (in_channels, *shape)
    state_dict['img_size'] = shape[0]
    state_dict['num_classes'] = num_classes

    # load weights
    model.load_state_dict(state_dict, strict=strict)
    del state_dict

    # return model on device
    return model


def load_b0(
    model_dir: pathlib.Path,
    model_name: str,
    device: torch.nn.Module = torch.device('cpu'),
    **kw
) -> torch.nn.Module:
    model = get_b0(**kw).to(device)
    resume_model_file = model_dir / model_name / 'model' / 'best_model.pt.tar'
    checkpoint = torch.load(resume_model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'model {model_name} loaded')
    print(f'model {model_name} loaded')
    return model
