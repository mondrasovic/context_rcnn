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

from torch import optim


def make_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY
    )

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.LR_SCHED.STEP_SIZE,
        gamma=cfg.LR_SCHED.GAMMA
    )
    
    return lr_scheduler
