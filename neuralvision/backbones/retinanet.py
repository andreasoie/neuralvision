from typing import Any, Union

from neuralvision.backbones.resnet import ResNet
from neuralvision.backbones.utils import (OutputChannels, OutputFeatures,
                                          TensorTuple, dict_to_tensors,
                                          tensors_to_dict)
from torch import Tensor, rand
from torch.nn import Module
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class RetinaNet(Module):
    def __init__(
        self,
        output_channels: OutputChannels,
        image_channels: int,
        output_feature_sizes: OutputFeatures,
        fpn_out_channels: int = 256, # FPN Spesific?
        extra_blocks: Union[Any, None] = None,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_sizes = output_feature_sizes
        self.fpn_out_channels = fpn_out_channels
        self.extra_blocks = extra_blocks
        self.out_channels = [self.fpn_out_channels]*len(self.output_channels)

        self.backbone = ResNet(self.output_channels, self.image_channels, self.output_feature_sizes)
        self.fpn = FeaturePyramidNetwork(self.output_channels, self.fpn_out_channels, self.extra_blocks)
    
    def forward(self, x: Tensor) -> TensorTuple:
        out = self.backbone(x)
        out = tensors_to_dict(out)
        out = self.fpn(out)
        out = dict_to_tensors(out)
        return out



if __name__ == "__main__":

    batch_size = 10
    image_channels = 3
    image_size = (128, 1024)
    output_channels = [128, 256, 128, 128, 64, 64]
    output_feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]

    test_img = rand(batch_size, image_channels, *image_size)  

    retina = RetinaNet(
        output_channels=output_channels,
        image_channels=image_channels,
        output_feature_sizes=output_feature_sizes
    )

    features = retina(test_img)

    for fet in features:
        print(fet.shape)
