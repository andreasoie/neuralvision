import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .ssd_multibox_loss import hard_negative_mining

class FocalLoss2(nn.Module):
    def __init__(self, gamma, alphas):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alphas)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)                        # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
       
        return loss.sum()

class FocalLossy(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def one_hot_embedding(labels, num_classes):
            return torch.eye(num_classes)[labels.data.cpu()]
            
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
        # self.foccy = FocalLossy(len(alphas))
        self.foccy = FocalLoss2(0.25, alphas)

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


        # M0
        # with torch.no_grad():
        #     to_log = -F.log_softmax(confs, dim=1)[:, 0] # torch.Size([32, 65440])
        #     mask = hard_negative_mining(to_log, gt_labels, 3.0)
        # classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        # classification_loss = classification_loss[mask].sum()

        # # M1
        # logpt = F.log_softmax(confs, dim=1)
        # pt = torch.exp(logpt)
        # logpt = (1-pt)**self.gamma * logpt
        # loss1 = F.nll_loss(logpt, gt_labels, None)

        # # M2
        # ce_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        # p_t = torch.exp(-ce_loss)
        # loss2 = (1 - p_t)**self.gamma * ce_loss
        # if self.reduction == "mean":
        #     loss2 = loss2.mean()
        # elif self.reduction == "sum":
        #     loss2 = loss2.sum()

        # # M3
        # ce_loss = F.cross_entropy(confs, gt_labels, reduction="mean", weight=None)
        # pt = torch.exp(-ce_loss)
        # loss3 = ((1 - pt) ** self.gamma * ce_loss).mean()

        # M4
        classification_loss = self.foccy(confs, gt_labels)
        
    
        # self.idx += 1
        # if self.idx % 10 == 0:
        #     print("**"*30)
        #     print("FocalLoss 0: ", classification_loss)
        #     print("FocalLoss 1: ", loss1)
        #     print("FocalLoss 2: ", loss2)
        #     print("FocalLoss 3: ", loss3)
        #     print("FocalLoss 4: ", loss4)
        #     print("**"*30)

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

        # TEST = False
        # if TEST:

        #     print("Max label: ", gt_labels[0, :].max())

        #     valid_mask = gt_labels >= 0
        #     print("valid_mask: ", valid_mask.shape)

        #     pk = to_log
        #     log_pk = torch.log(to_log)
        #     yk = F.one_hot(gt_labels[valid_mask], num_classes=8+1)[:, :-1]
        #     yk = yk.reshape(32, 65440, 8)
        #     print(f"pk: {pk.shape} with e.g. {pk[0, :4]}")
        #     print(f"pk log: {log_pk.shape}")
        #     print(f"yk: {yk.shape} with example: {yk[0, 0, :]}")
        #     print(f"ak: {self.alphas.shape}")
        #     # tmp = ak    * (1- pk)^2 * yk * log(pk)
        #     # tmp = 9x32 * 32x65440 ^2  * 32x65440x9 * 32x65440
        #     # tmp_i = 1x9 * 1x1^2  * 1x9 * 1x1
        #     # x = torch.pow()

        #     tmp = torch.pow(1-pk, self.gamma)
        #     print("tmp: ", tmp.shape)
        #     print("AK has shape and dim: ", self.alphas.shape, self.alphas.dim(), " and tmp has shape and dim: ", tmp.shape, tmp.dim())
        #     tmp = torch.matmul(self.alphas, tmp)
        #     print("tmp: ", tmp.shape)
        #     tmp = torch.matmul(tmp, yk)
        #     print("tmp: ", tmp.shape)
        #     tmp = torch.matmul(tmp, log_pk)
        #     print("tmp: ", tmp.shape)
        #     tmp = -torch.sum(tmp)
        #     print(f"tmp: {tmp.shape}")


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
