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

import math
import sys

import torch

from .utils import (
    MetricLogger, SmoothedValue, reduce_dict, to_device, save_checkpoint
)
from .eval import evaluate


def do_train(
    model,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    n_epochs,
    start_epoch=1,
    data_loader_va=None,
    eval_freq=1,
    checkpoints_dir_path=None,
    checkpoint_save_freq=None,
    print_freq=10,
    log_dir=None
):
    for epoch in range(start_epoch, n_epochs + 1):
        _train_one_epoch(
            model, optimizer, data_loader, device, epoch, n_epochs, print_freq
        )
        lr_scheduler.step()

        if checkpoints_dir_path and ((epoch % checkpoint_save_freq) == 0):
            save_checkpoint(
                checkpoints_dir_path, model, optimizer, lr_scheduler, epoch
            )
        
        if (data_loader_va is not None) and ((epoch % eval_freq) == 0):
            evaluate(model, data_loader_va, device=device)


def _train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    n_epochs,
    print_freq,
    scaler=None
):
    model.train()

    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = f"Epoch: [{epoch}/{n_epochs}]"

    for images, targets in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        images = to_device(images, device)
        targets = [
            {key:to_device(val, device) for key, val in target.items()}
            for target in targets
        ]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return metric_logger
