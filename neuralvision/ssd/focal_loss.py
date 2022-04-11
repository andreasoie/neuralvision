import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import RetinaNet
from torchvision.models.detection import RetinaNet
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int, long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, loss, labels):
        if loss.dim()>2:
            loss = loss.view(loss.size(0), loss.size(1), -1)  # N,C,H,W => N,C,H*W
            loss = loss.transpose(1, 2)    # N,C,H*W => N,H*W,C
            loss = loss.contiguous().view(-1,loss.size(2))   # N,H*W,C => N*H*W,C
        labels = labels.view(-1, 1)

        logpt = F.log_softmax(loss)
        logpt = logpt.gather(1, labels)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != loss.data.type():
                self.alpha = self.alpha.type_as(loss.data)
            at = self.alpha.gather(0, labels.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()