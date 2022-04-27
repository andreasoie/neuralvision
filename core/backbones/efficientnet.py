import torch
from torch import nn
import torchvision


def load_efficientnet(coef: int, pretrained: bool):
    """
    Loads the EfficientNet model with the given coefficients.
    :param coef: The coefficients of the model.
    :param pretrained: Whether to load the pretrained weights.
    :return: The model.
    """
    model = None
    model_name = f"efficientnet_b{coef}"

    if model_name == "efficientnet_b0":
        from torchvision.models.efficientnet import efficientnet_b0

        model = efficientnet_b0(pretrained=pretrained)
    elif model_name == "efficientnet_b1":
        from torchvision.models.efficientnet import efficientnet_b1

        model = efficientnet_b1(pretrained=pretrained)
    elif model_name == "efficientnet_b2":
        from torchvision.models.efficientnet import efficientnet_b2

        model = efficientnet_b2(pretrained=pretrained)
    elif model_name == "efficientnet_b3":
        from torchvision.models.efficientnet import efficientnet_b3

        model = efficientnet_b3(pretrained=pretrained)
    elif model_name == "efficientnet_b4":
        from torchvision.models.efficientnet import efficientnet_b4

        model = efficientnet_b4(pretrained=pretrained)
    elif model_name == "efficientnet_b5":
        from torchvision.models.efficientnet import efficientnet_b5

        model = efficientnet_b5(pretrained=pretrained)
    elif model_name == "efficientnet_b6":
        from torchvision.models.efficientnet import efficientnet_b6

        model = efficientnet_b6(pretrained=pretrained)
    elif model_name == "efficientnet_b7":
        from torchvision.models.efficientnet import efficientnet_b7

        model = efficientnet_b7(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


class EfficientNet(nn.Module):
    def __init__(self, coef: int, pretrained: bool):
        super().__init__()
        model = load_efficientnet(coef, pretrained)

        self.features = model.features
        output_channels_map = {
            "efficientnet-b0": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            "efficientnet-b1": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            "efficientnet-b2": [32, 16, 24, 48, 88, 120, 208, 352, 1408],
        }
        # self.out_channels = output_channels_map[f"efficientnet-b{coef}"]
        self.out_channels = output_channels_map[f"efficientnet-b{coef}"]

    def forward(self, x):
        features = []
        for feature in self.features:
            x = feature(x)
            features.append(x)
        return tuple(features)


def run_test(model, img):
    img_copy = img.clone()
    print("Input: ", img_copy.shape)
    for i, feat in enumerate(model.features):
        img_copy = feat(img_copy)
        print(f"{i+1}     - {img_copy.shape}")


def test_all():
    for coef in [0, 1, 2, 3, 4, 5, 6, 7]:
        model = EfficientNet(coef, True)
        print(f"EfficientNet-{coef}")
        run_test(model, torch.randn(5, 3, 128, 1024))
        print()


"""

OUTPUT DIMENSIONS for img_size = (128, 1024)

EfficientNet-0
Input:  torch.Size([5, 3, 128, 1024])
1     - torch.Size([5, 32, 64, 512])
2     - torch.Size([5, 16, 64, 512])
3     - torch.Size([5, 24, 32, 256])
4     - torch.Size([5, 40, 16, 128])
5     - torch.Size([5, 80, 8, 64])
6     - torch.Size([5, 112, 8, 64])
7     - torch.Size([5, 192, 4, 32])
8     - torch.Size([5, 320, 4, 32])
9     - torch.Size([5, 1280, 4, 32])

EfficientNet-1
Input:  torch.Size([5, 3, 128, 1024])
1     - torch.Size([5, 32, 64, 512])
2     - torch.Size([5, 16, 64, 512])
3     - torch.Size([5, 24, 32, 256])
4     - torch.Size([5, 40, 16, 128])
5     - torch.Size([5, 80, 8, 64])
6     - torch.Size([5, 112, 8, 64])
7     - torch.Size([5, 192, 4, 32])
8     - torch.Size([5, 320, 4, 32])
9     - torch.Size([5, 1280, 4, 32])

EfficientNet-2
Input:  torch.Size([5, 3, 128, 1024])
1     - torch.Size([5, 32, 64, 512])
2     - torch.Size([5, 16, 64, 512])
3     - torch.Size([5, 24, 32, 256])
4     - torch.Size([5, 48, 16, 128])
5     - torch.Size([5, 88, 8, 64])
6     - torch.Size([5, 120, 8, 64])
7     - torch.Size([5, 208, 4, 32])
8     - torch.Size([5, 352, 4, 32])
9     - torch.Size([5, 1408, 4, 32])

EfficientNet-3
Input:  torch.Size([5, 3, 128, 1024])
1     - torch.Size([5, 40, 64, 512])
2     - torch.Size([5, 24, 64, 512])
3     - torch.Size([5, 32, 32, 256])
4     - torch.Size([5, 48, 16, 128])
5     - torch.Size([5, 96, 8, 64])
6     - torch.Size([5, 136, 8, 64])
7     - torch.Size([5, 232, 4, 32])
8     - torch.Size([5, 384, 4, 32])
9     - torch.Size([5, 1536, 4, 32])

EfficientNet-4
Input:  torch.Size([5, 3, 128, 1024])
1     - torch.Size([5, 48, 64, 512])
2     - torch.Size([5, 24, 64, 512])
3     - torch.Size([5, 32, 32, 256])
4     - torch.Size([5, 56, 16, 128])
5     - torch.Size([5, 112, 8, 64])
6     - torch.Size([5, 160, 8, 64])
7     - torch.Size([5, 272, 4, 32])
8     - torch.Size([5, 448, 4, 32])
9     - torch.Size([5, 1792, 4, 32])

"""
