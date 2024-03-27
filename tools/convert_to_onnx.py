from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import torch
import torch.nn as nn

import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--arch', type=str, help='backbone architecture')


args = parser.parse_args()


class ConvertModel(nn.Module):
    def __init__(self, model):
        super(ConvertModel, self).__init__()
        self.model = model

    def forward(self, template, search):
        zf = self.model.backbone(template)
        if cfg.ADJUST.ADJUST:
            zf = self.model.neck(zf)
        xf = self.model.backbone(search)
        if cfg.ADJUST.ADJUST:
            xf = self.model.neck(xf)
        cls, loc = self.model.rpn_head(zf, xf)
        return cls, loc


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval()
    model = ConvertModel(model)
    model.eval()


    if args.arch == 'rpn_alexnet':
        x = torch.randn(1, 3, 125, 125)
        z = torch.randn(1, 3, 287, 287)
    elif args.arch == 'rpn_mobilenetv2':
        x = torch.randn(1, 3, 127, 127)
        z = torch.randn(1, 3, 255, 255)
    else:
        raise ValueError('arch not supported')

    torch.onnx.export(
        model,
        (x, z),
        f'{args.arch}.onnx',
        export_params=True,
        do_constant_folding=False,
        opset_version=16,  # Adjust the opset version as necessary
    )


if __name__ == '__main__':
    main()
