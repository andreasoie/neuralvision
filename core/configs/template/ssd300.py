import os

import torch
import torchvision
from core.configs.dir_utils import get_dataset_dir, get_output_dir

from core.backbones.basic import BasicModel
from core.datasets_classes.mnist_detection_dataset import MNISTDetectionDataset
from core.helpers import batch_collate, batch_collate_val
from core.ssd.anchor_boxes import AnchorBoxes
from core.ssd.ssd import SSD300
from core.ssd.ssd_multibox_loss import SSDMultiboxLoss
from core.tops.config.lazy import LazyCall as L
from core.transforms.gpu_transforms import Normalize
from core.transforms.target_transform import GroundTruthBoxesToAnchors
from core.transforms.transform import ToTensor
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

MNIST_OD_DATASET = "datasets/mnist_object_detection"

train = dict(
    batch_size=32,
    amp=True,  # Automatic mixed precision
    log_interval=20,
    seed=0,
    epochs=50,
    _output_dir=get_output_dir(),
    imshape=(128, 1024),
    image_channels=3,
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
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
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
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
