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

import abc
import dataclasses
import itertools
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
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
        image_idx: int
    
    def __init__(
        self,
        root_path,
        subset='train',
        *,
        past_context=0,
        future_context=0,
        context_stride=1,
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
            transforms (Callable, optional): Transformation to apply to
                individual frames. Beware that if context is required, some
                transformations may be nonsensical. Defaults to None.
        """
        self._context_rel_idxs = _calc_context_rel_idxs(
            past_context, future_context, context_stride
        )

        self._global_to_local_seq_image_idxs = []
        self._seq_image_paths = []
        self._seq_boxes = []

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
                'context_images': A list of tensors of contextual images
                    (including the center image).
        """
        seq_box_idx = self._global_to_local_seq_image_idxs[idx]
        seq_idx, center_image_idx = seq_box_idx.seq_idx, seq_box_idx.image_idx

        image_file_paths = self._seq_image_paths[seq_idx]
        abs_context_idxs = np.clip(
            self._context_rel_idxs + center_image_idx, 0,
            len(image_file_paths) - 1
        )
        
        center_image = None
        context_images = []
        prev_idx = -1

        for context_idx in abs_context_idxs:            
            if context_idx != prev_idx:
                image_file_path = image_file_paths[context_idx]
                image = Image.open(image_file_path)
            if context_idx == center_image_idx:
                center_image = image
            context_images.append(image)

            prev_idx = context_idx
        
        assert center_image is not None

        image_id = torch.as_tensor([idx], dtype=torch.int64)
        boxes = self._seq_boxes[seq_idx][center_image_idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        is_crowd = torch.zeros_like(labels)

        target = {
            'image_id': image_id,
            'boxes': boxes,
            'area': areas,
            'labels': labels,
            'iscrowd': is_crowd,
            'context_images': context_images,
        }

        if self.transforms is not None:
            center_image, target = self.transforms(center_image, target)

        return center_image, target

    def __len__(self):
        """Returns the length of the dataset. It represents the number of
        images (frames) of the entire dataset.

        Returns:
            int: Dataset length.
        """
        return len(self._global_to_local_seq_image_idxs)

    def _init_data_indices(self, root_path, subset):
        """Initializes data indices to faster access. It reads image (frame)
        file names as well as their corresponding bounding boxes for faster
        access later on.

        Args:
            root_path (str): UA-DETRAC dataset root path.
            subset (str): Whether to read 'train' or 'test' data subset.
        """
        images_dir, annos_dir = self._deduce_images_and_annos_paths(
            root_path, subset
        )

        for seq_idx, seq_dir in enumerate(images_dir.iterdir()):
            xml_file_name = seq_dir.stem + '_v3.xml'
            xml_file_path = str(annos_dir / xml_file_name)

            image_boxes_map = dict(self._iter_seq_boxes(xml_file_path))

            seq_image_file_paths = []
            seq_image_boxes = []

            image_idx_gen = itertools.count()

            for image_num, image_file_path in self._iter_seq_image_file_paths(
                seq_dir
            ):
                boxes = image_boxes_map.get(image_num)
                if boxes is not None:
                    seq_image_file_paths.append(image_file_path)
                    seq_image_boxes.append(boxes)

                    image_idx = next(image_idx_gen)
                    seq_boxes_idx = self._SeqBoxesIndex(seq_idx, image_idx)
                    self._global_to_local_seq_image_idxs.append(seq_boxes_idx)
            
            self._seq_image_paths.append(seq_image_file_paths)
            self._seq_boxes.append(seq_image_boxes)

    @staticmethod
    def _deduce_images_and_annos_paths(root_path, subset):
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

        images_idr = root_dir / ('Insight-MVT_Annotation_' + subset)
        annos_dir = root_dir / 'DETRAC_public' / ('540p-' + subset)

        return images_idr, annos_dir
    
    @staticmethod
    def _iter_seq_image_file_paths(seq_dir):
        """Iterates over image file names for a specific sequence from the
        UA-DETRAC dataset.

        Args:
            seq_dir (pathlib.Path): Sequence directory path.

        Yields:
            Tuple[int, str]: Tuple containing image (frame) number and
            the corresponding file path.
        """
        image_num_path_pairs = [
            (int(p.stem[-5:]), str(p)) for p in seq_dir.iterdir()
        ]
        yield from iter(sorted(image_num_path_pairs))

    @staticmethod
    def _iter_seq_boxes(xml_file_path):
        """Iterates over a sequence of bounding boxes contained within a
        specific XML file corresponding to some sequence from the UA-DETRAC
        dataset.

        Args:
            xml_file_path (str): Sequence specification XML file path.

        Yields:
            Tuple[int, List[Tuple[float, float, float, float]]]: A tuple
            containing the frame number and the list of bounding boxes in a
            xyxy format.
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
                boxes.append(box)
            
            yield frame_num, boxes



def transform_center_and_context_images(image, target, transform):
    image = transform(image)
    
    context_images = target.get('context_images') or []
    for i, context_image in enumerate(context_images):
        context_images[i] = transform(context_image)
    
    return image, target


class TransformWithContext(abc.ABC):
    @abc.abstractmethod
    def __call__(self, image, target):
        pass


class ToTensorWithContext(TransformWithContext):
    def __init__(self):
        self.to_tensor = T.ToTensor()
    
    def __call__(self, image, target):
        image, target = transform_center_and_context_images(
            image, target, self.to_tensor
        )
        return image, target


class ColorJitterWithContext(TransformWithContext):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target):
        image, target = transform_center_and_context_images(
            image, target, self.color_jitter
        )
        return image, target


class ComposeTransforms(TransformWithContext):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def collate_context_images_batch(batch):
    images, targets = map(list, zip(*batch))
    return images, targets


def make_transforms(cfg, train=True):
    to_tensor = ToTensorWithContext()
    transforms = [to_tensor]
    
    if train:
        color_jitter = ColorJitterWithContext(
            cfg.DATASET.AUG.BRIGHTNESS, cfg.DATASET.AUG.CONTRAST,
            cfg.DATASET.AUG.SATURATION, cfg.DATASET.AUG.HUE
        )
        transforms.append(color_jitter)
    
    transforms = ComposeTransforms(transforms)

    return transforms


def make_dataset(cfg, *, train=True):
    transforms = make_transforms(cfg, train)

    subset = 'train' if train else 'test'

    dataset = UADetracContextDetectionDataset(
        cfg.DATASET.ROOT_PATH, subset, past_context=cfg.DATASET.PAST_CONTEXT,
        future_context=cfg.DATASET.FUTURE_CONTEXT,
        context_stride=cfg.DATASET.CONTEXT_STRIDE, transforms=transforms
    )

    return dataset


def make_data_loader(
    cfg,
    dataset,
    train=True,
    collate_fn=collate_context_images_batch
):
    pin_memory = torch.cuda.is_available()
    shuffle = cfg.DATA_LOADER.SHUFFLE if train else False
    
    data_loader = DataLoader(
        dataset, batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.N_WORKERS,
        collate_fn=collate_fn, pin_memory=pin_memory
    )

    return data_loader


def _calc_context_rel_idxs(past_context, future_context, stride):
    assert past_context >= 0
    assert future_context >= 0
    assert stride > 0

    past_idxs = -np.flip(np.arange(stride, past_context + 1, stride))
    center_idx = np.asarray([0])
    future_idxs = np.arange(stride, future_context + 1, stride)

    idxs = np.concatenate((past_idxs, center_idx, future_idxs))
    
    return idxs


if __name__ == '__main__':
    import functools

    import cv2 as cv

    from config import cfg

    
    def tensor_to_cv_image(image_tensor, max_size=None):
        image = image_tensor.cpu().numpy()  # [C,H,W]
        image = np.transpose(image, (1, 2, 0))  # [H,W,C]
        image = image[..., ::-1]

        if max_size is not None:
            height, width, _ = image.shape
            max_side = max(height, width)
            
            if max_side > max_size:
                scale = max_size / max_side
                image = cv.resize(
                    image, None, fx=scale, fy=scale,
                    interpolation=cv.INTER_CUBIC
                )
        
        return image
    

    class ImageBatchVisualizer:
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
        
        def preview_batch_images(self, images, targets):
            image_rows = []

            _to_cv_image = functools.partial(
                tensor_to_cv_image, max_size=self.max_size
            )

            for _, target in zip(images, targets):
                image_cols = list(map(_to_cv_image, target['context_images']))
                image_cols_merged = np.hstack(image_cols)
                image_rows.append(image_cols_merged)
            
            image_final = np.vstack(image_rows)

            cv.imshow(self.win_name, image_final)
            key = cv.waitKey(0) & 0xff

            return key != ord(self.quit_key)
        
        @staticmethod
        def _calc_temporal_win_size(past_context, future_context, stride):
            # TODO Implement better, purely arithmetic-based solution.
            return len(_calc_context_rel_idxs(
                past_context, future_context, stride
            ))

    dataset = make_dataset(cfg)
    data_loader = make_data_loader(cfg, dataset)

    n_batches_shown = 4

    with ImageBatchVisualizer(cfg, max_size=400) as visualizer:
        for images, targets in itertools.islice(data_loader, n_batches_shown):
            if not visualizer.preview_batch_images(images, targets):
                break
