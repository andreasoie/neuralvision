from typing import List, Tuple

import torch
from torch import nn
from torchvision.models import resnet18

# from torchvision.models.resnet import Bottleneck
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# from torchvision.models.efficientnet import EfficientNet

OutputFeatures = List[List[int]]
OutputChannels = List[int]
ImageChannels = int


class ResNetConfig:
    def __init__(self, resnet: nn.Sequential, output_channels: OutputChannels) -> None:
        self.resnet = resnet
        self.out_channels = output_channels

    def init_custom_tail(self) -> nn.Sequential:
        N_LAYERS = 6
        layers = [child for child in self.resnet.children()]
        layers = layers[:N_LAYERS]
        layers = nn.Sequential(*layers)
        # Update strides: from 2,2 to 1,1
        conv4_block1 = layers[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)
        return layers

    def init_custom_heads(self) -> nn.ModuleList:
        extra_layers = []
        inn_channels = self.out_channels[:-1]
        mid_channels = self.out_channels
        out_channels = self.out_channels[1:]
        channels = zip(inn_channels, mid_channels, out_channels)
        for ch_in, ch_mid, ch_out in channels:
            extra_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_mid, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ch_mid),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_mid,
                        ch_out,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.ModuleList(extra_layers)


class ResNet(nn.Module):
    def __init__(
        self,
        output_channels: OutputChannels,
        image_channels: ImageChannels,
        output_feature_sizes: OutputFeatures,
    ) -> None:
        super().__init__()
        self.out_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes

        resnet_cfg = ResNetConfig(resnet18(), output_channels)
        self.tail = resnet_cfg.init_custom_tail()
        self.heads = resnet_cfg.init_custom_heads()
        self.debug = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:

        x = self.tail(x)
        features = [x]
        for head in self.heads:
            x = head(x)
            features.append(x)

        return tuple(features)


if __name__ == "__main__":

    BS = 4
    image_channels = 3
    output_channels = [128, 256, 128, 128, 64, 64]
    feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]
    imshape = (128, 1024)
    random_image = torch.randn(BS, image_channels, *imshape)

    resnet = ResNet(output_channels, image_channels, feature_sizes)

    out = resnet(random_image)

    print("IMAGE: ", random_image.shape)
    for o in out:
        print("RESNET: ", o.shape)