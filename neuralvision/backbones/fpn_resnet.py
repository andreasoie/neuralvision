from typing import Any, List, Union

import torch
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from neuralvision.backbones.resnet import ResNet

OutputFeatures = List[List[int]]
OutputChannels = List[int]
ImageChannels = int


class FPNResnet(nn.Module):
    def __init__(
        self,
        output_channels: OutputChannels,
        image_channels: ImageChannels,
        output_feature_sizes: OutputFeatures,
        in_channels_list: OutputChannels,
        out_channels: int = 6,
        extra_blocks: Union[Any, None] = None,
    ) -> None:

        self.backbone = ResNet(output_channels, image_channels, output_feature_sizes)
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x = self.fpn(features)
        return x
