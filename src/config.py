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

from yacs.config import CfgNode as CN


_C = CN()

# ------------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.ROOT_PATH = '../../../datasets/UA-DETRAC'
_C.DATASET.SUBSET = 'train'

_C.DATASET.GROUP_HORIZONTAL_FLIP = None
_C.DATASET.IMG_MEAN = [0.485, 0.456, 0.406]
_C.DATASET.IMG_STD = [0.229, 0.224, 0.225]

_C.DATASET.PAST_CONTEXT = 1
_C.DATASET.FUTURE_CONTEXT = 1
_C.DATASET.CONTEXT_STRIDE = 1

# ------------------------------------------------------------------------------
_C.DATA_LOADER = CN()

_C.DATA_LOADER.BATCH_SIZE = 4
_C.DATA_LOADER.SHUFFLE = True
_C.DATA_LOADER.NUM_WORKERS = 2

# ------------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.NUM_CLASSES = 2  # 0 - background, 1 - vehicle

cfg = _C
