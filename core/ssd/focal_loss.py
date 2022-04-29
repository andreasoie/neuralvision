from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLossFunction(nn.Module):
    def __init__(self, gamma, alphas):
        super().__init__()
        self.gamma = gamma
        self.alpha = alphas

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha.type() != input.data.type():
            self.alpha = self.alpha.type_as(input.data)
        at = self.alpha.gather(0, target.data.view(-1))
        logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(
        self,
        anchors,
        alphas: List[Union[float, int]],
        gamma: float = 2.0,
        reduction: str = "sum",
    ):
        super().__init__()
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh
        self.sl1_loss = nn.SmoothL1Loss(reduction="none")
        self.anchors = nn.parameter.Parameter(
            anchors(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False
        )
        self.alphas = torch.Tensor(alphas)
        self.gamma = gamma
        self.reduction = reduction
        self.focal_loss = FocalLossFunction(self.gamma, self.alphas)

    def _loc_vec(self, loc):
        """
        Generate Location Vectors
        """
        gxy = (
            self.scale_xy
            * (loc[:, :2, :] - self.anchors[:, :2, :])
            / self.anchors[
                :,
                2:,
            ]
        )
        gwh = self.scale_wh * (loc[:, 2:, :] / self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(
        self,
        bbox_delta: torch.FloatTensor,
        confs: torch.FloatTensor,
        gt_bbox: torch.FloatTensor,
        gt_labels: torch.LongTensor,
    ):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        # bbox_delta: torch.Size([32, 4, 65440]) dtype= torch.float16
        # confs: torch.Size([32, 9, 65440]) dtype= torch.float16
        # gt_bbox: torch.Size([32, 4, 65440]) dtype= torch.float32
        # gt_labels: torch.Size([32, 65440]) dtype= torch.int64

        # reshape to [batch_size, 4, num_anchors]
        gt_bbox = gt_bbox.transpose(1, 2).contiguous()  # type: ignore

        classification_loss = self.focal_loss(confs, gt_labels)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]  # type: ignore
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0] / 4
        total_loss = regression_loss / num_pos + classification_loss / num_pos
        to_log = dict(
            regression_loss=regression_loss / num_pos,
            classification_loss=classification_loss / num_pos,
            total_loss=total_loss,
        )
        return total_loss, to_log
