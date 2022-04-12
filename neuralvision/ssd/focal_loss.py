import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, anchors, alphas: List[Union[float, int]], gamma: float = 2.0, reduction: str = "sum"):
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
        self.idx = 0

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
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # type: ignore

        
        # with torch.no_grad():
        #     to_log = -F.log_softmax(confs, dim=1)[:, 0] # torch.Size([32, 65440])
        # classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")

        # from collections import Counter

        # lab = gt_labels[0, :]

        # stuff = Counter(lab.detach().cpu().tolist())
        # for ob in stuff.items():
        #     print(ob)

        # # (0, 65365)
        # # (1, 71)
        # # (7, 4)

        logpt = F.log_softmax(confs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        classification_loss = F.nll_loss(logpt, gt_labels, None)

        ce_loss = F.cross_entropy(confs, gt_labels, weight=None, reduction="none")
        p_t = torch.exp(-ce_loss)
        lossy = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            lossy = lossy.mean()
        elif self.reduction == "sum":
            lossy = lossy.sum()

        self.idx += 1

        if self.idx % 10 == 0:
            print("**"*30)
            print("FocalLoss 1: ", classification_loss)
            print("FocalLoss 2: ", lossy)
            print("**"*30)

        classification_loss = lossy
        # ak: [32, 9]
        # pk: [32, 65440] (k is the softmax output for class k)
        # yk: [.., ..   ] (y is the ground truth one-hot encoded (1 if the ground truth correspond to class k else 0).)

        # yk[0] = [1, 65440] # one image, 65k boxes, one-hot encoded
        # One-Hot labels
        # gt_labels = 

        # pk: torch.Size([32, 65440])
        # yk: torch.Size([2094080, 9])
        # ak: torch.Size([9, 32])

        # Print example from gt_labels

        TEST = False
        if TEST:

            print("Max label: ", gt_labels[0, :].max())

            valid_mask = gt_labels >= 0
            print("valid_mask: ", valid_mask.shape)

            pk = to_log
            log_pk = torch.log(to_log)
            yk = F.one_hot(gt_labels[valid_mask], num_classes=8+1)[:, :-1]
            yk = yk.reshape(32, 65440, 8)
            print(f"pk: {pk.shape} with e.g. {pk[0, :4]}")
            print(f"pk log: {log_pk.shape}")
            print(f"yk: {yk.shape} with example: {yk[0, 0, :]}")
            print(f"ak: {self.alphas.shape}")
            # tmp = ak    * (1- pk)^2 * yk * log(pk)
            # tmp = 9x32 * 32x65440 ^2  * 32x65440x9 * 32x65440
            # tmp_i = 1x9 * 1x1^2  * 1x9 * 1x1
            # x = torch.pow()

            tmp = torch.pow(1-pk, self.gamma)
            print("tmp: ", tmp.shape)
            print("AK has shape and dim: ", self.alphas.shape, self.alphas.dim(), " and tmp has shape and dim: ", tmp.shape, tmp.dim())
            tmp = torch.matmul(self.alphas, tmp)
            print("tmp: ", tmp.shape)
            tmp = torch.matmul(tmp, yk)
            print("tmp: ", tmp.shape)
            tmp = torch.matmul(tmp, log_pk)
            print("tmp: ", tmp.shape)
            tmp = -torch.sum(tmp)
            print(f"tmp: {tmp.shape}")


        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask] # type: ignore
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
