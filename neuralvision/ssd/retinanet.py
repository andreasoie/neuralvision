import math
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


def create_subnet_stem(channels: int) -> nn.Sequential:
    """ initializes the RetinaNet stem-subnetworks """
    layers = []
    for _ in range(4):
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def create_subnet(in_chan: int, out_chan: int, init: str = "default") -> nn.Sequential:
    # build subnet stem
    stem = create_subnet_stem(in_chan)
    # build subnet head
    head = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)

    if init == "classification":
        # See Focal Loss paper for details
        for layer in stem.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        # Final output with its own initialization
        torch.nn.init.normal_(head.weight, std=0.01)
        torch.nn.init.constant_(head.bias, -math.log((1 - 0.01) / 0.01))
        stem.add_module("convhead", head)

    elif init == "regression":
        # See Focal Loss paper for details
        for layer in stem.children():
            
            if isinstance(layer, nn.Conv2d):
                l1 = layer.weight
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)
                l2 = layer.weight

                # torch check if alike
                torch.testing.assert_allclose(l1, l2, rtol=0, atol=0)

        # Final output with its own initialization
        torch.nn.init.normal_(head.weight, std=0.01)
        torch.nn.init.zeros_(head.bias)
        stem.add_module("convhead", head)
        
    elif init == "default":
        # According to SSD paper
        stem.add_module("convhead", head)
        for param in stem.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    else:
        raise NotImplementedError(f"Unknown subnet init: {init}")

    return nn.Sequential(*stem)

class RetinaNet(nn.Module):
    """ Implements the Retina network """
    def __init__(self, feature_extractor: nn.Module, anchors, loss_objective: Any, num_classes: int) -> None:
        super().__init__()

        self.feature_extractor: nn.Module = feature_extractor
        self.loss_func: Any = loss_objective
        self.num_classes: int = num_classes
        self.regression_heads: List[nn.Sequential] = []
        self.classification_heads: List[nn.Sequential] = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.anchor_encoder = AnchorEncoder(anchors)

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
            init: regression => initializes regression focal style weights
            init: classification => initializes regression focal style weights
            init: default => initializes default SSD style weights"""

            reg_channels: Tuple[int, int] = out_channel, num_boxes * 4
            cls_channels: Tuple[int, int] = out_channel, num_boxes * self.num_classes

            reg_head = create_subnet(*reg_channels, init = "regression")
            cls_head = create_subnet(*cls_channels, init = "classification")

            self.regression_heads.append(reg_head.to(self.device))
            self.classification_heads.append(cls_head.to(self.device))

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
