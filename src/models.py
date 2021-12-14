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

import itertools
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor, TwoMLPHead
)
from torchvision.models.detection.roi_heads import (
    RoIHeads, fastrcnn_loss, keypointrcnn_loss, maskrcnn_loss,
    keypointrcnn_inference, maskrcnn_inference
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign


class AttentionEmbMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        n_hidden_size = out_dim * 2

        self.fc1: nn.Module = nn.Linear(in_dim, n_hidden_size, bias=False)
        self.relu2: nn.Module = nn.ReLU(inplace=True)
        self.fc3 = nn.Module = nn.Linear(n_hidden_size, out_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes attention embedding. It maps input or context features into
        an intermediate embedding space that will be used as part of the
        attention mechanism.

        Args:
            features (torch.Tensor): Feature tensor of shape [N,C], where C is
                the input dimension (no. of channels) specified in the
                constructor.

        Returns:
            torch.Tensor: Feature embeddings of shape [N,E], where E is the
                output (embedding) dimension specified in the constructor.
        """
        x = features
        x = self.fc1(x)  # [N,H], where H is the hidden layer size.
        x = self.relu2(x)  # [N,H]
        x = self.fc3(x)  # [N,E]

        return x


class ContextAttention(nn.Module):
    def __init__(
        self,
        n_feature_channels: int,
        *,
        query_key_dim: int = 256,
        value_dim: int = 256,
        softmax_temperature: float = 0.01,
    ) -> None:
        super().__init__()

        self.query_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, query_key_dim
        )
        self.key_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, query_key_dim
        )
        self.value_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, value_dim
        )
        self.final_mapper: nn.Module = AttentionEmbMapper(
            value_dim, n_feature_channels
        )

        self.softmax_scale: float = (
            1.0 / (softmax_temperature * (n_feature_channels ** 0.5))
        )
    
    def forward(
        self,
        central_features: torch.Tensor,
        context_features: torch.Tensor
    ) -> torch.Tensor:
        """Computes attention bias for the given input and context features.

        Args:
            central_features (torch.Tensor): Central frame features of shape
                [N,C,S,S], where N is the no. of proposals per image, N is the
                batch size (no. of images) multiplied by the no. of proposals
                per image.
            context_features (torch.Tensor): Context frames features of shape
                [T,C,S,S], where T is the temporal window size (1 + no. of
                images in the past and future) multiplied by the batch size
                (no. of images).

        Returns:
            torch.Tensor: Attention feature bias of shape [N,C].
        """
        # Apply global average pooling (GAP).
        central_features = torch.mean(central_features, dim=(2, 3))  # [N,C]
        context_features = torch.mean(context_features, dim=(2, 3))  # [T,C]

        queries = self.query_mapper(central_features)  # [N,D1]
        keys = self.key_mapper(context_features)   # [T,D1]
        values = self.value_mapper(context_features)  # [T,D2]

        queries = F.normalize(queries, dim=1)  # [N,D1]
        keys = F.normalize(keys, dim=1)  # [T,D1]

        weights = torch.matmul(queries, keys.T)  # [N,T]
        weights = F.softmax(weights * self.softmax_scale, dim=1)  # [N,T]

        values_weighted = torch.matmul(weights, values)  # [N,D2]

        attention_biases = self.final_mapper(values_weighted)  # [N,C]

        return attention_biases


# Code adapted from
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
class RoiHeadsWithContext(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Context
        n_feature_channels,
        query_key_dim=None,
        value_dim=None,
        softmax_temperature=0.01,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction, bbox_reg_weights,
            score_thresh, nms_thresh, detections_per_img, mask_roi_pool,
            mask_head, mask_predictor, keypoint_roi_pool, keypoint_head,
            keypoint_predictor
        )

        if query_key_dim is None:
            query_key_dim = n_feature_channels
        
        if value_dim is None:
            value_dim = n_feature_channels
        
        self.st_attention = ContextAttention(
            n_feature_channels, query_key_dim=query_key_dim,
            value_dim=value_dim, softmax_temperature=softmax_temperature
        )

    def forward(
        self,
        features,
        proposals,
        image_shapes,
        context_features,
        context_proposals,
        context_image_shapes,
        targets=None,
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t['boxes'].dtype in floating_point_types, "target boxes must of float type"
                assert t['labels'].dtype == torch.int64, "target labels must of int64 type"
                if self.has_keypoint():
                    assert t['keypoints'].dtype == torch.float32, "target keypoints must of float type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        context_box_features = self.box_roi_pool(
            context_features, context_proposals, context_image_shapes
        )

        attention_bias = self.st_attention(box_features, context_box_features)
        box_features += attention_bias[..., None, None]
        
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                'loss_classifier': loss_classifier,
                'loss_box_reg': loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        'boxes': boxes[i],
                        'labels': labels[i],
                        'scores': scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p['boxes'] for p in result]
            if self.training:
                assert matched_idxs is not None
                # During training, only focus on positive boxes.
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t['masks'] for t in targets]
                gt_labels = [t['labels'] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels,
                    pos_matched_idxs
                )
                loss_mask = {'loss_mask': rcnn_loss_mask}
            else:
                labels = [r['labels'] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r['masks'] = mask_prob

            losses.update(loss_mask)

        # Keep none checks in if conditional so torchscript will conditionally
        # compile each branch.
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p['boxes'] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t['keypoints'] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints,
                    pos_matched_idxs
                )
                loss_keypoint = {'loss_keypoint': rcnn_loss_keypoint}
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(
                    keypoints_probs, kp_scores, result
                ):
                    r['keypoints'] = keypoint_prob
                    r['keypoints_scores'] = kps

            losses.update(loss_keypoint)

        return result, losses


# Code adapted from
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
class ContextRCNN(nn.Module):
    """
    Implements Context R-CNN.

    The input to the model is expected to be a list of tensors, each of shape
    [C, H, W], one for each image, and should be in 0-1 range. Different images
    can have different sizes.

    The behavior of the model changes depending if it is in training or
    evaluation mode.

    During training, the model expects both the input tensors, as well as a
    targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """

    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters.
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters.
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters.
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Context parameters
        query_key_dim=None,
        value_dim=None,
        softmax_temperature=0.01,
    ):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        self.backbone = backbone

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is " +
                    "specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor is " +
                    "not specified"
                )

        out_channels = self.backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh,
            rpn_bg_iou_thresh, rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7,
                sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2, representation_size
            )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        self.roi_heads = RoiHeadsWithContext(
            box_roi_pool, box_head, box_predictor, box_fg_iou_thresh,
            box_bg_iou_thresh, box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights, box_score_thresh, box_nms_thresh,
            box_detections_per_img, n_feature_channels=out_channels,
            query_key_dim=query_key_dim, value_dim=value_dim,
            softmax_temperature=softmax_temperature
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )

        self._has_warned = False

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the
                image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the
                losses. During testing, it returns list[BoxList] contains
                additional fields like `scores`, `labels` and `mask`
                (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a "
                            f"tensor of shape [N, 4], got {boxes.shape}."
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of "
                        f"type Tensor, got {type(boxes)}."
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        context_images = self._join_context_images(targets)

        images, targets = self.transform(images, targets)
        context_images, _ = self.transform(context_images)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target['boxes']
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # Print the first degenerate box.
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and "
                        f"width. Found invalid box {degen_bb} for target at "
                        f"index {target_idx}."
                    )
        
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        context_features = self.backbone(context_images.tensors)
        if isinstance(context_features, torch.Tensor):
            context_features = OrderedDict([('0', context_features)])
        
        proposals, proposal_losses = self.rpn(images, features, targets)
        with _evaluating(self.rpn):  # Avoid computing RPN loss.
            context_proposals, _ = self.rpn(context_images, context_features)

        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, context_features,
            context_proposals, context_images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in "
                    "scripting"
                )
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections
    
    def _join_context_images(self, targets):
        context_images = list(
            itertools.chain.from_iterable(
                target['context_images'] for target in targets
            )
        )
        return context_images


def make_faster_rcnn_model(cfg):
    """Makes a Faster R-CNN model with ResNet-50 FPN backbone pre-trained on
    MS COCO dataset with the classification head replaced with a different
    number of classes specified using the cfg.MODEL.N_CLASSES attribute.

    Args:
        cfg (CfgNode): YACS configuration.

    Returns:
        nn.Module: Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained_backbone=cfg.MODEL.PRETRAINED_BACKBONE,
        trainable_backbone_layers=cfg.MODEL.TRAINABLE_BACKBONE_LAYERS
    )
    num_classes = cfg.MODEL.N_CLASSES
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model


def make_context_rcnn_model(cfg):
    """[summary]

    Args:
        cfg (CfgNode): YACS configuration.

    Returns:
        nn.Module: Context R-CNN model.
    """
    pretrained_backbone = cfg.MODEL.PRETRAINED_BACKBONE
    trainable_backbone_layers = cfg.MODEL.TRAINABLE_BACKBONE_LAYERS
    
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3
    )
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained_backbone,
        trainable_layers=trainable_backbone_layers
    )

    model = ContextRCNN(
        backbone, cfg.MODEL.N_CLASSES, min_size=540, max_size=960,
        query_key_dim=cfg.MODEL.ATTENTION.QUERY_KEY_DIM,
        value_dim=cfg.MODEL.ATTENTION.VALUE_DIM,
        softmax_temperature=cfg.MODEL.ATTENTION.SOFTMAX_TEMP
    )

    return model


def make_object_detection_model(cfg):
    model_name = cfg.MODEL.NAME
    
    if model_name == 'FasterRCNN':
        return make_faster_rcnn_model(cfg)
    elif model_name == 'ContextRCNN':
        return make_context_rcnn_model(cfg)
    else:
        raise ValueError("unrecognized model name: " + model_name)


def _validate_trainable_layers(
    pretrained: bool,
    trainable_backbone_layers: Optional[int],
    max_value: int,
    default_value: int,
) -> int:
    # Don't freeze any layers if pretrained model or backbone is not used.
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to "
                f"True, falling back to trainable_backbone_layers={max_value} "
                "so that all layers are trainable"
            )
        trainable_backbone_layers = max_value

    # By default freeze first blocks.
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value

    return trainable_backbone_layers


@contextmanager
def _evaluating(model: nn.Module):
    """Temporarily switch to evaluation mode.

    Args:
        model (nn.Module): Model to modify the state of.

    Yields:
        nn.Module: The provided model.
    """
    is_train = model.training
    try:
        model.eval()
        yield model
    finally:
        if is_train:
            model.train()
