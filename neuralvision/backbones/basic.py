from typing import List, Tuple

import torch

from .utils import validate_backbone_features


class BasicModel(torch.nn.Module):
    def __init__(
        self,
        output_channels: List[int],
        image_channels: int,
        output_feature_sizes: List[List[int]],
    ):
        super().__init__()
        self.out_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes
        self.debug = True

        self.layers = self.get_layers()

    def get_layers(self):
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.image_channels, 32, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                64, self.out_channels[0], kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.out_channels[0], 128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                128, self.out_channels[1], kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.out_channels[1], 256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                256, self.out_channels[2], kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.out_channels[2], 128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                128, self.out_channels[3], kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.out_channels[3], 128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                128, self.out_channels[4], kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.out_channels[4], 128, kernel_size=2, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                128, self.out_channels[5], kernel_size=2, stride=2, padding=0
            ),
            torch.nn.ReLU(),
        )
        return [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        ]

    def forward(self, x):
        out_features = []
        for layer in self.layers:
            x = layer(x)
            out_features.append(x)

        if self.debug:
            validate_backbone_features(
                self.output_feature_shape, self.out_channels, out_features
            )
        return tuple(out_features)


if __name__ == "__main__":

    ich = 3
    feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]
    output_channels = [128, 256, 128, 128, 64, 64]
    model = BasicModel(output_channels, ich, feature_sizes)

    test_img = torch.randn(1, ich, 128, 1024)

    out = model(test_img)
    print("Image: ", test_img.shape)
    for o in out:
        print("  {o.name}: ", o.shape)
