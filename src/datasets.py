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
import functools
import itertools

import numpy as np
import torch

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T


class UADetracContextDetectionDataset(torch.utils.data.Dataset):
    """UA-DETRAC detection dataset with the additional capability of adding
    contextual images. If no context is specified, this class produces a dataset
    representation that is tantamount to that which torchvision object detection
    modules expect.
    """

    @dataclasses.dataclass(frozen=True)
    class _SeqBoxesIndex:
        """An auxiliary classs to store sequence index and image index within
        that sequence.
        """
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
        """Constructor.

        Args:
            root_path (str): Path to the UA-DETRAC dataset.
            subset (str, optional): Data subset ('train' or 'test').
                Defaults to 'train'.
            past_context (int, optional): A non-negative integer specifying the
                number of frames in the past. Defaults to 0.
            future_context (int, optional): A non-negative integer specifying
                the number of frames in the future. Defaults to 0.
            context_stride (int, optional): A positive integer representing the
                stride when traversing the past as well as future contextual
                frames. Defaults to 1.
            group_horizontal_flip (float, optional): If set, it should be a
                float number in the <0, 1> interval representing the probability
                of horizontally flipping all the images within a single group,
                i.e., the center + the contextual images. Defaults to None.
            transforms (Callable, optional): Transformation to apply to
                individual frames. Beware that if context is required, some
                transformations may be nonsensical. Defaults to None.
        """
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
        """Retrieves a random sample from the dataset.

        Args:
            idx (int): Data sample index.

        Returns:
            Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]: Returns a data
            sample consisting of a center image in a tensor format, and target
            specification as a dictionary with the following content:
                'boxes': A Nx4 tensor of boxes in xyxy format.
                'labels': A N, tensor of labels (0 indicates background).
                'context_images': A list of tensors of contextual images.
        """
        seq_box_idx = self._global_to_local_seq_img_idxs[idx]
        seq_idx, center_img_idx = seq_box_idx.seq_idx, seq_box_idx.img_idx

        img_file_paths = self._seq_img_paths[seq_idx]
        abs_context_idxs = np.clip(
            self._context_rel_idxs + center_img_idx, 0, len(img_file_paths) - 1
        )

        def _read_img(idx):
            img_file_path = img_file_paths[idx]
            img = Image.open(img_file_path)

            if self.transforms is not None:
                img = self.transforms(img)
            
            return img
        
        center_img = _read_img(center_img_idx)
        context_imgs = []
        prev_idx = center_img_idx

        for context_idx in abs_context_idxs:            
            if prev_idx != context_idx:
                img = _read_img(context_idx)
            context_imgs.append(img)

            prev_idx = context_idx

        boxes = self._seq_boxes[seq_idx][center_img_idx]
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'context_imgs': context_imgs,
        }

        return center_img, target

    def __len__(self):
        """Returns the length of the dataset. It represents the number of
        images (frames) of the entire dataset.

        Returns:
            int: Dataset length.
        """
        return len(self._global_to_local_seq_img_idxs)

    def _init_data_indices(self, root_path, subset):
        """Initializes data indices to faster access. It reads image (frame)
        file names as well as their corresponding bounding boxes in a tensor
        format for faster access later on.

        Args:
            root_path ([type]): UA-DETRAC dataset root path.
            subset ([type]): Whether to read 'train' or 'test' data subset.
        """
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
        """Iterates over image file names for a specific sequence from the
        UA-DETRAC dataset.

        Args:
            seq_dir (pathlib.Path): Sequence directory path.

        Yields:
            Tuple[int, str]: Yield tuples containing image (frame) number and
            the corresponding file path.
        """
        for img_file in seq_dir.iterdir():
            img_num = int(img_file.stem[-5:])
            img_file_path = str(img_file)

            yield img_num, img_file_path

    @staticmethod
    def _iter_seq_boxes(xml_file_path):
        """Iterates over a sequence of bounding boxes contained within a
        specific XML file corresponding to some sequence from the UA-DETRAC
        dataset.

        Args:
            xml_file_path (str): Sequence specification XML file path.

        Yields:
            Tuple[int, List[torch.Tensor]]: A tuple containing the frame number
            and the list of bounding boxes in a tensor xyxy format.
        """
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
                box = torch.as_tensor(box, dtype=torch.float32)
                boxes.append(box)
            
            yield frame_num, boxes


def collate_context_imgs_batch(batch):
    imgs, targets = map(list, zip(*batch))
    return imgs, targets


def make_transforms(cfg, train=True):
    transforms = [T.ToTensor()]
    
    if train:
        color_jitter = T.ColorJitter(
            cfg.DATASET.AUG.BRIGHTNESS, cfg.DATASET.AUG.CONTRAST,
            cfg.DATASET.AUG.SATURATION, cfg.DATASET.AUG.HUE
        )
        transforms.append(color_jitter)
    
    transforms = T.Compose(transforms)

    return transforms


def make_uadetrac_dataset(cfg, train=True):
    transforms = make_transforms(cfg, train)

    dataset = UADetracContextDetectionDataset(
        cfg.DATASET.ROOT_PATH, cfg.DATASET.SUBSET,
        past_context=cfg.DATASET.PAST_CONTEXT,
        future_context=cfg.DATASET.FUTURE_CONTEXT,
        context_stride=cfg.DATASET.CONTEXT_STRIDE,
        group_horizontal_flip=cfg.DATASET.AUG.GROUP_HORIZONTAL_FLIP,
        transforms=transforms
    )

    return dataset


def make_data_loader(cfg, dataset, collate_fn=collate_context_imgs_batch):
    data_loader = DataLoader(
        dataset, batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return data_loader


def _calc_context_rel_idxs(past_context, future_context, stride):
    assert past_context >= 0
    assert future_context >= 0
    assert stride > 0

    past_idxs = np.arange(-past_context, 0, stride)
    future_idxs = np.arange(1, future_context + 1, stride)

    idxs = np.concatenate((past_idxs, future_idxs))
    
    return idxs


if __name__ == '__main__':
    import itertools

    import cv2 as cv
    import torch.nn.functional as F

    from config import cfg

    
    def tensor_to_cv_img(img_tensor, max_size=None):
        img = img_tensor.cpu().numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]
        img = img[..., ::-1]

        if max_size is not None:
            height, width, _ = img.shape
            max_side = max(height, width)
            
            if max_side > max_size:
                scale = max_size / max_side
                img = cv.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC
                )
        
        return img
    

    class ImgBatchVisualizer:
        def __init__(
            self,
            cfg,
            *,
            max_size=None,
            win_name='Batch Preview',
            quit_key='q'
        ):
            self.past_context = cfg.DATASET.PAST_CONTEXT
            self.temporal_win_size = self._calc_temporal_win_size(
                self.past_context, cfg.DATASET.FUTURE_CONTEXT,
                cfg.DATASET.CONTEXT_STRIDE
            )

            self.max_size = max_size
            self.win_name = win_name
            self.quit_key = quit_key
        
        def __enter__(self):
            cv.namedWindow(self.win_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            cv.destroyWindow(self.win_name)
        
        def preview_batch_imgs(self, imgs, targets):
            img_rows = []

            _to_cv_img = functools.partial(
                tensor_to_cv_img, max_size=self.max_size
            )

            for img_tensor, target in zip(imgs, targets):
                center_img = _to_cv_img(img_tensor)
                img_cols = list(map(_to_cv_img, target['context_imgs']))
                center_img = _to_cv_img(img_tensor)
                img_cols.insert(self.past_context, center_img)

                img_cols_merged = np.hstack(img_cols)
                img_rows.append(img_cols_merged)
            
            img_final = np.vstack(img_rows)

            cv.imshow(self.win_name, img_final)
            key = cv.waitKey(0) & 0xff

            return key != ord(self.quit_key)
        
        @staticmethod
        def _calc_temporal_win_size(past_context, future_context, stride):
            # TODO Implement better, purely arithmetic-based solution.
            return 1 + len(_calc_context_rel_idxs(
                past_context, future_context, stride
            ))

    dataset = make_uadetrac_dataset(cfg)
    data_loader = make_data_loader(cfg, dataset)

    n_batches_shown = 4

    with ImgBatchVisualizer(cfg, max_size=400) as visualizer:
        for imgs, targets in itertools.islice(data_loader, n_batches_shown):
            if not visualizer.preview_batch_imgs(imgs, targets):
                break
