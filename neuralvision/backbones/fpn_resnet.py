import collections
from typing import Any, List, Union, OrderedDict

import torch
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from resnet import ResNet

TensorDict = OrderedDict[str, torch.Tensor]
OutputFeatures = List[List[int]]
OutputChannels = List[int]
ImageChannels = int


def tensor_to_dict(feature_maps: torch.Tensor) -> TensorDict:
    feature_dict: TensorDict = collections.OrderedDict()
    for i, feature_map in enumerate(feature_maps):
        feature_dict[f"feat{i}"] = feature_map
    return feature_dict

class FPNResnet(nn.Module):
    def __init__(
        self,
        output_channels: OutputChannels,
        image_channels: ImageChannels,
        output_feature_sizes: OutputFeatures,
        out_channels: int = 256, # FPN Spesific?
        extra_blocks: Union[Any, None] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNet(output_channels, image_channels, output_feature_sizes)
        self.fpn = FeaturePyramidNetwork(output_channels, out_channels, extra_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = tensor_to_dict(features)
        features = self.fpn(features)
        return features



if __name__ == "__main__":

    batch_size = 10
    image_channels = 3
    image_size = (128, 1024)
    output_channels = [128, 256, 128, 128, 64, 64]
    output_feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]

    test_img = torch.rand(batch_size, image_channels, *image_size)

    retina = FPNResnet(
        output_channels=output_channels,
        image_channels=image_channels,
        output_feature_sizes=output_feature_sizes
    )

    features = retina(test_img)
    for k, v in features.items():
        print(f"{k} {v.shape}")

    # random_img = torch.rand(2, 3, 128, 1024)
    # inplanes = 64 // 8 # = 8

    # model = FPNResnet(
    # rfpn = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

    # test_img = torch.rand(2, 3, 128, 1024)

    # out = rfpn(test_img)

    # for k, v in out.items():
    #     print(f"{k} {v.shape}")