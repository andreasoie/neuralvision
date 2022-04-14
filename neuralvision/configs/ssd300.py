import os

import torch
import torchvision
from neuralvision.configs.dir_utils import get_dataset_dir, get_output_dir

from neuralvision.backbones.basic import BasicModel
from neuralvision.datasets_classes.mnist_detection_dataset import MNISTDetectionDataset
from neuralvision.helpers import batch_collate, batch_collate_val
from neuralvision.ssd.anchor_boxes import AnchorBoxes
from neuralvision.ssd.ssd import SSD300
from neuralvision.ssd.ssd_multibox_loss import SSDMultiboxLoss
from neuralvision.tops.config.lazy import LazyCall as L
from neuralvision.transforms.gpu_transforms import Normalize
from neuralvision.transforms.target_transform import GroundTruthBoxesToAnchors
from neuralvision.transforms.transform import ToTensor
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

MNIST_OD_DATASET = "datasets/mnist_object_detection"

train = dict(
    batch_size=32,
    amp=True,  # Automatic mixed precision
    log_interval=20,
    seed=0,
    epochs=2,
    _output_dir=get_output_dir(),
    imshape=(128, 1024),
    image_channels=3,
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[
        [16, 16],
        [32, 32],
        [48, 48],
        [64, 64],
        [86, 86],
        [128, 128],
        [128, 400],
    ],
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2,
)

backbone = L(BasicModel)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
)

loss_objective = L(SSDMultiboxLoss)(anchors="${anchors}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=10 + 1,  # Add 1 for background
)

optimizer = L(torch.optim.SGD)(
    lr=5e-3,
    momentum=0.9,
    weight_decay=0.0005,
)

schedulers = dict(
    linear=L(LinearLR)(start_factor=0.1, end_factor=1, total_iters=500),
    multistep=L(MultiStepLR)(milestones=[], gamma=0.1),
)

data_train = dict(
    dataset=L(MNISTDetectionDataset)(
        data_dir=get_dataset_dir(f"{MNIST_OD_DATASET}/train"),
        is_train=True,
        transform=L(torchvision.transforms.Compose)(
            transforms=[
                L(ToTensor)(),
                # ToTensor has to be applied before conversion to anchors.
                # GroundTruthBoxesToAnchors assigns each ground truth to anchors, \
                # required to compute loss in training.
                L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
            ]
        ),
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}",
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        batch_size="${...train.batch_size}",
        collate_fn=batch_collate,
        drop_last=True,
    ),
    # GPU transforms can heavily speedup data augmentations.
    gpu_transform=L(torchvision.transforms.Compose)(
        transforms=[
            L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # Normalize has to be appliedafter ToTensor
            # (GPU transform is always after CPU)
        ]
    ),
)
data_val = dict(
    dataset=L(MNISTDetectionDataset)(
        data_dir=get_dataset_dir(f"{MNIST_OD_DATASET}/val"),
        is_train=False,
        transform=L(torchvision.transforms.Compose)(transforms=[L(ToTensor)()]),
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}",
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        batch_size="${...train.batch_size}",
        collate_fn=batch_collate_val,
    ),
    gpu_transform=L(torchvision.transforms.Compose)(
        transforms=[
            L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
)

label_map = {0: "background", **{i + 1: str(i) for i in range(10)}}
