import torch
import torchvision


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).float().view(1, len(mean), 1, 1)
        self.std = torch.tensor(std).float().view(1, len(mean), 1, 1)

    @torch.no_grad()
    def forward(self, batch):
        self.mean = self.mean.to(batch["image"].device)
        self.std = self.std.to(batch["image"].device)
        batch["image"] = (batch["image"] - self.mean) / self.std
        return batch


class ColorJitter(torchvision.transforms.ColorJitter):
    @torch.no_grad()
    def forward(self, batch):
        batch["image"] = super().forward(batch["image"])
        return batch

class RandomAdjustSharpness(torchvision.transforms.RandomAdjustSharpness):
    """
    - sharpness_factor (float) – How much to adjust the sharpness.
    Can be any non negative number.
    0 gives a blurred image,
    1 gives the original image,
    2 increases the sharpness by a factor of 2.

    - p (float) – probability of the image being sharpened. Default value is 0.5
    """
    def __init__(self, sharpness_factor: float = 1, p: float = 0.5):
        super().__init__(sharpness_factor, p)

    @torch.no_grad()
    def forward(self, batch):
        batch["image"] = super().forward(batch["image"])
        return batch
