import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.ops.focal_loss import sigmoid_focal_loss

# import one_hot

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    # print(f"loss: {loss.shape} dtype: {loss.dtype} sample data {loss[0]}")
    # print(f"labels: {labels.shape} dtype: {labels.dtype} sample data {labels[0]}")

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    targets = targets.unsqueeze(0)
    tf = F.one_hot(targets)
    print(f"Inputs: {inputs.shape} dtype= {inputs.dtype}")
    print(f"Targets: {targets.shape} dtype= {targets.dtype}")
    print(f"TF: {tf.shape} dtype= {tf.dtype}")
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def multiclass_focal_log_loss(y_pred, y_true, class_weights = None, alpha = 0.5, gamma = 2):
    print(f"y_pred: {y_pred.shape} dtype= {y_pred.dtype} sample value: {y_pred[0]}")
    print(f"y_true: {y_true.shape} dtype= {y_true.dtype} sample value: {y_true[0]}")
    eps = 1e-12
    # If actual value is true, keep pt value as y_pred otherwise (1-y_pred)
    pt = torch.where(y_true == 1, y_pred, 1-y_pred)
    # If actual value is true, keep alpha_t value as alpha otherwise (1-alpha)
    alpha_t = torch.where(y_true == 1, alpha, 1-alpha)
    # Clip values below epsilon and above 1-epsilon
    pt = torch.clip(pt, eps, 1-eps)

    # FL = -alpha_t(1-pt)^gamma log(pt)
    # focal_loss = -torch.mean(torch.multiply(torch.multiply(alpha_t, torch.power(1-pt,gamma)), torch.log(pt)), axis=0)
    focal_loss = - alpha_t * torch.pow(1-pt, gamma) * torch.log(pt)
    if class_weights is None:
        focal_loss = torch.mean(focal_loss)
    else:
        focal_loss = torch.sum(torch.multiply(focal_loss, class_weights))
    return focal_loss


def focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2, eps=1e-12):
    bce = F.binary_cross_entropy(y_true, y_pred)
    y_pred = torch.clip(y_pred, eps, 1.- eps)
    p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
    
    alpha_factor = 1
    modulating_factor = 1

    alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
    modulating_factor = torch.pow((1-p_t), gamma)

    # compute the final loss and return
    return torch.mean(alpha_factor * modulating_factor * bce)

class SSDMultiboxLoss(nn.Module):
    """
    Implements the loss as the sum of the followings:
    1. Confidence Loss: All labels, with hard negative mining
    2. Localization Loss: Only on positive labels
    Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction="none")
        self.anchors = nn.Parameter(
            anchors(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False
        )
        self.foc = focal_loss(gamma=2)


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
        # reshape to [batch_size, 4, num_anchors]
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # type: ignore  
        
        with torch.no_grad():
            to_log = -F.log_softmax(confs, dim=1)[:, 0]
            mask = hard_negative_mining(to_log, gt_labels, 3.0)
        classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        classification_loss = classification_loss[mask].sum()

        class_loss = self.foc(confs, gt_labels)
        print(f"class_loss: {class_loss}")

        # classification and regression loss
        # print(f"confs: {confs.shape} dtype= {bbox_delta.dtype} sample value: {confs[0]}")
        # print(f"gt_labels: {gt_labels.shape} dtype= {bbox_delta.dtype} sample value: {gt_labels[0]}")
        # classification_loss = sigmoid_focal_loss(confs, gt_labels)
        # classification_loss = multiclass_focal_log_loss(confs, gt_labels)

        # focal_crossentropy_loss = focal_crossentropy(gt_labels, confs)
        # print(f"focal_crossentropy_loss: {focal_crossentropy_loss}")

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

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.  # type: ignore
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)  # type: ignore
        alpha = alpha.to(device=device, dtype=dtype)  # type: ignore

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl