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
from .datasets import make_dataset, make_data_loader
from .models import make_object_detection_model
from .optim import make_optimizer, make_lr_scheduler
from .train import do_train
from .eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faster RCNN, Context RCNN: training/evaluation script.",
    )
    
    parser.add_argument(
        '-c', '--config', help="path to the YAML configuration file"
    )
    parser.add_argument(
        '--checkpoints-dir', help="path to the directory containing checkpoints"
    )
    parser.add_argument(
        '--log-dir', help="Directory containing logs + tensorboard writer data."
    )
    parser.add_argument(
        '--checkpoint-file',
        help="specific checkpoint file to restore the model training or "
        "evaluating from"
    )

    parser.add_argument(
        'opts', metavar='OPTIONS', nargs=argparse.REMAINDER,
        help="overwriting the default YAML configuration"
    )
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    dataset_tr = make_dataset(cfg, train=True)
    dataset_va = make_dataset(cfg, train=False)
    data_loader_tr = make_data_loader(cfg, dataset_tr)
    data_loader_va = make_data_loader(cfg, dataset_va)

    model = make_object_detection_model(cfg).to(device)

    optimizer = make_optimizer(cfg, model)
    lr_scheduler = make_lr_scheduler(cfg, optimizer)

    n_epochs = cfg.TRAIN.N_EPOCHS
    eval_freq = cfg.TRAIN.EVAL_FREQ
    checkpoint_save_freq = cfg.TRAIN.CHECKPOINT_SAVE_FREQ
    print_freq = cfg.TRAIN.PRINT_FREQ

    do_train(
        model, optimizer, lr_scheduler, data_loader_tr, device, n_epochs,
        data_loader_va, eval_freq, args.checkpoints_dir, args.checkpoint_file,
        checkpoint_save_freq, print_freq, args.log_dir
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
