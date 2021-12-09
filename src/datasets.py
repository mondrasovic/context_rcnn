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

import dataclasses
import itertools

import numpy as np
import torch

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T


class UADetracDetectionDataset(torch.utils.data.Dataset):
    @dataclasses.dataclass(frozen=True)
    class _SeqBoxesIndex:
        seq_idx: int
        img_idx: int
    
    def __init__(
        self,
        root_path,
        subset='train',
        *,
        past_context=0,
        future_context=0,
        context_stride=1,
        group_horizontal_flip=None,
        transforms=None
    ):
        self._context_rel_idxs = _calc_context_rel_idxs(
            past_context, future_context, context_stride
        )

        self._global_to_local_seq_img_idxs = []
        self._seq_img_paths = []
        self._seq_boxes = []

        self.group_horizontal_flip = group_horizontal_flip
        self.transforms = transforms

        self._init_data_indices(root_path, subset)

    def __getitem__(self, idx):
        seq_box_idx = self._global_to_local_seq_img_idxs[idx]

        img_file_paths = self._seq_img_paths[seq_box_idx.seq_idx]
        img_boxes = self._seq_boxes[seq_box_idx.seq_idx]

        abs_idxs = np.clip(
            self._context_rel_idxs + seq_box_idx.img_idx, 0,
            len(img_file_paths) - 1
        )

        imgs, boxes = [], []

        prev_idx = -1
        img = None

        for idx in abs_idxs:
            img_file_path, curr_boxes = img_file_paths[idx], img_boxes[idx]

            if prev_idx != idx:
                img = Image.open(img_file_path)
            
            if self.transforms is not None:
                img = self.transforms(img)
            
            imgs.append(img)
            boxes.append(curr_boxes)

            prev_idx = idx
        
        imgs = torch.stack(imgs)

        return imgs

    def __len__(self):
        """Returns the length of the dataset. It represents the number of
        images (frames) of the entire dataset.

        Returns:
            int: Dataset length.
        """
        return len(self._global_to_local_seq_img_idxs)

    def _init_data_indices(self, root_path, subset):
        imgs_dir, annos_dir = self._deduce_imgs_and_annos_paths(
            root_path, subset
        )

        for seq_idx, seq_dir in enumerate(imgs_dir.iterdir()):
            if seq_idx > 2:  # TODO Remove this ugly break.
                break

            xml_file_name = seq_dir.stem + '_v3.xml'
            xml_file_path = str(annos_dir / xml_file_name)

            img_files_iter = self._iter_seq_img_file_paths(seq_dir)
            img_boxes_iter = self._iter_seq_boxes(xml_file_path)
            data_iter = itertools.zip_longest(img_files_iter, img_boxes_iter)

            seq_img_file_paths = []
            seq_img_boxes = []

            for img_idx, (files_iter_data, boxes_iter_data) in enumerate(
                data_iter
            ):
                assert files_iter_data is not None
                assert boxes_iter_data is not None

                img_num_1, img_file_path = files_iter_data
                img_num_2, boxes = boxes_iter_data
                assert img_num_1 == img_num_2

                seq_img_file_paths.append(img_file_path)
                seq_img_boxes.append(boxes)

                seq_boxes_idx = self._SeqBoxesIndex(seq_idx, img_idx)
                self._global_to_local_seq_img_idxs.append(seq_boxes_idx)
            
            self._seq_img_paths.append(seq_img_file_paths)
            self._seq_boxes.append(seq_img_boxes)

    @staticmethod
    def _deduce_imgs_and_annos_paths(root_path, subset):
        """Deduces paths for images and annotations. It returns the root path
        that contains all the sequences belonging to the specific subset.

        Args:
            root_path (str): Root directory path to the UA-DETRAC dataset.
            subset (str): Data subset type ('train' or 'test').

        Returns:
            Tuple[pathlib.Path, pathlib.Path]: Directory paths for images and
            annotations.
        """
        assert subset in ('train', 'test')

        subset = subset.capitalize()
        root_dir = Path(root_path)

        imgs_dir = root_dir / ('Insight-MVT_Annotation_' + subset)
        annos_dir = root_dir / 'DETRAC_public' / ('540p-' + subset)

        return imgs_dir, annos_dir
    
    @staticmethod
    def _iter_seq_img_file_paths(seq_dir):
        for img_file in seq_dir.iterdir():
            img_num = int(img_file.stem[-5:])
            img_file_path = str(img_file)

            yield img_num, img_file_path

    @staticmethod
    def _iter_seq_boxes(xml_file_path):
        tree = ElementTree.parse(xml_file_path)
        root = tree.getroot()

        for frame in root.findall('./frame'):
            frame_num = int(frame.attrib['num'])
            boxes = []

            for target in frame.findall('.//target'):
                box_attr = target.find('box').attrib
                
                x = float(box_attr['left'])
                y = float(box_attr['top'])
                w = float(box_attr['width'])
                h = float(box_attr['height'])

                box = (x, y, x + w, y + h)
                boxes.append(box)
            
            yield frame_num, boxes


def collate_context_imgs_batch(batch):
    return torch.cat(batch, dim=0)


def make_uadetrac_dataset(cfg):
    to_tensor = T.ToTensor()
    normalize = T.Normalize(cfg.DATASET.IMG_MEAN, cfg.DATASET.IMG_STD)
    transforms = T.Compose([to_tensor, normalize])

    dataset = UADetracDetectionDataset(
        cfg.DATASET.ROOT_PATH, cfg.DATASET.SUBSET,
        past_context=cfg.DATASET.PAST_CONTEXT,
        future_context=cfg.DATASET.FUTURE_CONTEXT,
        context_stride=cfg.DATASET.CONTEXT_STRIDE,
        group_horizontal_flip=cfg.DATASET.GROUP_HORIZONTAL_FLIP,
        transforms=transforms
    )

    return dataset


def make_data_loader(cfg, dataset, collate_fn=collate_context_imgs_batch):
    data_loader = DataLoader(
        dataset, batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS, collate_fn=collate_fn
    )

    return data_loader


def _calc_context_rel_idxs(past_context, future_context, context_stride):
    assert past_context >= 0
    assert future_context >= 0
    assert context_stride > 0

    past_idxs = np.arange(-past_context, 0, context_stride)
    center_idx = np.asarray([0])
    future_idxs = np.arange(1, future_context + 1, context_stride)

    idxs = np.concatenate((past_idxs, center_idx, future_idxs))
    
    return idxs


if __name__ == '__main__':
    import cv2 as cv
    import torch.nn.functional as F

    from config import cfg

    class ImgBatchVisualizer:
        def __init__(
            self,
            cfg,
            *,
            max_size=None,
            win_name='Batch Preview',
            quit_key='q'
        ):
            self.img_mean = cfg.DATASET.IMG_MEAN
            self.img_std = cfg.DATASET.IMG_STD

            context_size = cfg.DATASET.PAST_CONTEXT + cfg.DATASET.FUTURE_CONTEXT
            self.temporal_win_size = 1 + context_size

            self.max_size = max_size
            self.win_name = win_name
            self.quit_key = quit_key
        
        def __enter__(self):
            cv.namedWindow(self.win_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            cv.destroyWindow(self.win_name)

        def show_imgs(self, imgs_tensor):
            *_, height, width = imgs_tensor.shape

            if self.max_size is not None:
                max_side = max(height, width)
                if max_side > self.max_size:
                    scale = self.max_size / max_side
                    imgs_tensor = F.interpolate(
                        imgs_tensor, scale_factor=scale, mode='bicubic',
                        align_corners=True
                    )
            
            imgs = imgs_tensor.cpu().numpy()  # [B,C,H,W]
            imgs = np.transpose(imgs, (0, 2, 3, 1))  # [B,H,W,C]
            imgs = imgs[..., ::-1]
            imgs = (imgs * self.img_std) + self.img_mean

            n_imgs, height, width, n_channels = imgs.shape
            assert (n_imgs % self.temporal_win_size) == 0

            imgs = imgs.reshape(
                n_imgs // self.temporal_win_size, self.temporal_win_size,
                height, width, n_channels
            )  # [B/G,G,H,W,C]

            img_rows = []
            for imgs_group in imgs:
                img_cols = np.hstack(imgs_group)
                img_rows.append(img_cols)
            img_final = np.vstack(img_rows)

            cv.imshow(self.win_name, img_final)
            key = cv.waitKey(0) & 0xff

            return key != ord(self.quit_key)

    dataset = make_uadetrac_dataset(cfg)
    data_loader = make_data_loader(cfg, dataset)

    n_batches_shown = 4

    with ImgBatchVisualizer(cfg, max_size=400) as visualizer:
        for batch_idx, imgs in enumerate(data_loader, start=1):
            if batch_idx > n_batches_shown:
                break

            if not visualizer.show_imgs(imgs):
                break
