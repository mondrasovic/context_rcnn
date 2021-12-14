#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import argparse
import sys

import torch

from .config import cfg
from .datasets import make_uadetrac_dataset, make_data_loader
from .models import make_object_detection_model
from .optim import make_optimizer, make_lr_scheduler
from .train import train_one_epoch
from .eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faster RCNN, Context RCNN: training/evaluation script.",
    )
    
    parser.add_argument(
        '-c', '--config', help="path to the YAML configuration file"
    )
    parser.add_argument(
        'opts', metavar='OPTIONS', nargs=argparse.REMAINDER,
        help="overwriting the default YAML configuration"
    )
    
    args = parser.parse_args()

    return args


def train(device):
    dataset = make_uadetrac_dataset(cfg)
    data_loader = make_data_loader(cfg, dataset)
    data_loader_va = make_data_loader(cfg, dataset)

    model = make_object_detection_model(cfg).to(device)

    optimizer = make_optimizer(cfg, model)
    lr_scheduler = make_lr_scheduler(cfg, optimizer)

    model.train()

    num_epochs = cfg.TRAIN.N_EPOCHS
    print_freq = cfg.TRAIN.PRINT_FREQ

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=print_freq
        )
        lr_scheduler.step()
        # evaluate(model, data_loader_va, device=device)


def main():
    args = parse_args()

    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    train(device)

    return 0


if __name__ == '__main__':
    sys.exit(main())
