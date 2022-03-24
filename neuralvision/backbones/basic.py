from typing import List, Tuple

import torch


def validate_backbone_features(
    output_feature_size: List[Tuple[int]],
    output_channels: List[int],
    output_features: list,
):
    for idx, feature in enumerate(output_features):
        out_channel = output_channels[idx]
        h, w = output_feature_size[idx]
        expected_shape = (out_channel, h, w)

        err_msg1 = f"Expected shape: {expected_shape}, \
                        got: {feature.shape[1:]} at output IDX: {idx}"
        assert feature.shape[1:] == expected_shape, err_msg1

    err_msg2 = f"Expected that the length of the outputted features to be: \
        {len(output_feature_size)}, but it was: {len(output_features)}"
    assert len(output_features) == len(output_feature_size), err_msg2


class BasicModel(torch.nn.Module):
    def __init__(
        self,
        output_channels: List[int],
        image_channels: int,
        output_feature_sizes: List[Tuple[int]],
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
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
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
                self.out_channels[4], 128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                128, self.out_channels[5], kernel_size=3, stride=1, padding=0
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
