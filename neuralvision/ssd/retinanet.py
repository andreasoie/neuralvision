import math
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


def apply_weight_init(param, weight_style: str):
    assert isinstance(param, torch.nn.modules.conv.Conv2d)
    if weight_style == "default":
        # Seee SSD paper
        nn.init.xavier_uniform_(param.weight)
        # the bias is zero I assume
    elif weight_style == "classification":
        # See Focal Loss paper
        torch.nn.init.normal_(param.weight, std=0.01)
        torch.nn.init.constant_(param.bias, -math.log((1 - 0.01) / 0.01))
    elif weight_style == "regression":
        # See Focal Loss paper
        torch.nn.init.normal_(param.weight, std=0.01)
        torch.nn.init.zeros_(param.bias)
    else:
        raise NotImplementedError(f"Unknown weight_style: {weight_style}")

def create_subnet_stem(channels: int, weight_style: str) -> nn.Sequential:
    """ initializes the RetinaNet stem-subnetworks """
    layers = []
    for _ in range(4):
        _conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        _relu = nn.ReLU(inplace=True)
        apply_weight_init(_conv, weight_style)
        layers.append(_conv)
        layers.append(_relu)
    return nn.Sequential(*layers)

def create_subnet_head(in_chan: int, out_chan: int, weight_style: str) -> nn.Conv2d:
    stem_head = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
    apply_weight_init(stem_head, weight_style)
    return stem_head

def create_subnet(in_chan: int, out_chan: int, head_weight_style: str = "default") -> nn.Sequential:
    stem_weight_style = "default" # Don't change focal init I guess (default => SSD)
    # build subnet stem with initialized weights
    stem = create_subnet_stem(in_chan, stem_weight_style)
    # build subnet head with initialized weights
    head = create_subnet_head(in_chan, out_chan, head_weight_style)
    stem.add_module("convhead", head)
    return stem

def create_singlenet(in_chan: int, out_chan: int, head_weight_style: str = "default"):
    conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
    apply_weight_init(conv, head_weight_style)
    return conv


class RetinaNet(nn.Module):
    """ Implements the Retina network """
    def __init__(self, feature_extractor: nn.Module, anchors, loss_objective: Any, num_classes: int, use_deeper_head: bool, use_weightstyle: bool) -> None:
        super().__init__()

        self.feature_extractor: nn.Module = feature_extractor
        self.loss_func: Any = loss_objective
        self.num_classes: int = num_classes
        self.use_deeper_head: bool = use_deeper_head
        self.use_weightstyle: bool = use_weightstyle

        self.anchor_encoder = AnchorEncoder(anchors)
        self.regression_heads: nn.ModuleList = nn.ModuleList()
        self.classification_heads: nn.ModuleList = nn.ModuleList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        Initialize output heads that are applied
        to each feature map from the feature_extractor.

        if n_boxes is consistent e.g. 6,6,6,6,.., 
        and out channels is consistent e.g. 256,256,256,256,..,
        we can initialize only a single reg and class head
        if not, we loop and use varying numb_boxes and channels
        NB: paper uses consistent - we don't ?? """
        for num_boxes, out_channel in zip(anchors.num_boxes_per_fmap, self.feature_extractor.out_channels):
            """Create RetinaNet Subnet Heads 
            weight_style: regression => initializes regression focal style weights
            weight_style: classification => initializes regression focal style weights
            weight_style: default => initializes default SSD style weights"""

            reg_channels: Tuple[int, int] = out_channel, num_boxes * 4
            reg_head_weight_style = "default"

            cls_channels: Tuple[int, int] = out_channel, num_boxes * self.num_classes
            cls_head_weight_style = "classification" if self.use_weightstyle else "default"

            if self.use_deeper_head:
                reg_net = create_subnet(*reg_channels, head_weight_style = reg_head_weight_style)
                cls_net = create_subnet(*cls_channels, head_weight_style = cls_head_weight_style)
            else:
                reg_net = create_singlenet(*reg_channels, head_weight_style = "default")
                cls_net = create_singlenet(*cls_channels, head_weight_style = "default")
               
            self.regression_heads.append(reg_net.to(self.device))
            self.classification_heads.append(cls_net.to(self.device))

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for i, x in enumerate(features):
            bbox_delta = self.regression_heads[i](x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads[i](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    def forward(self, img: torch.Tensor, **kwargs):
        """img: shape: NCHW"""
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)

    def forward_test(
        self,
        img: torch.Tensor,
        imshape=None,
        nms_iou_threshold=0.5,
        max_output=200,
        score_threshold=0.05,
    ):
        """
        img: shape: NCHW
        nms_iou_threshold, max_output is only
        used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx],
                confs[img_idx],
                nms_iou_threshold,
                max_output,
                score_threshold,
            )
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions


def filter_predictions(
    boxes_ltrb: torch.Tensor,
    confs: torch.Tensor,
    nms_iou_threshold: float,
    max_output: int,
    score_threshold: float,
):
    """
    boxes_ltrb: shape [N, 4]
    confs: shape [N, num_classes]
    """
    assert 0 <= nms_iou_threshold <= 1
    assert max_output > 0
    assert 0 <= score_threshold <= 1
    scores, category = confs.max(dim=1)

    # 1. Remove low confidence boxes / background boxes
    mask = (scores > score_threshold).logical_and(category != 0)
    boxes_ltrb = boxes_ltrb[mask]
    scores = scores[mask]
    category = category[mask]

    # 2. Perform non-maximum-suppression
    keep_idx = batched_nms(
        boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold
    )

    # 3. Only keep max_output best boxes
    # (NMS returns indices in sorted order, decreasing w.r.t. scores)
    keep_idx = keep_idx[:max_output]
    return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]
