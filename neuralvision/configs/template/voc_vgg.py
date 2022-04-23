# Inherit configs from the default ssd300
import os
import torch
import torchvision
from neuralvision.backbones.vgg import VGG
from neuralvision.datasets_classes.voc_dataset import VOCDataset
from neuralvision.tops.config.lazy import LazyConfig as L
from neuralvision.transforms.gpu_transforms import Normalize
from neuralvision.transforms.target_transform import GroundTruthBoxesToAnchors
from neuralvision.transforms.transform import (
    RandomHorizontalFlip,
    RandomSampleCrop,
    Resize,
    ToTensor,
)
from neuralvision.configs.template.ssd300 import (
    anchors,
    data_train,
    data_val,
    loss_objective,
    model,
    optimizer,
    schedulers,
    train,
)
from dir_utils import get_dataset_dir

VOCDEVKIT_DATASET_DIR = "datasets/VOCdevkit"

# Keep the model, except change the backbone and number of classes
model.feature_extractor = L(VGG)()
model.num_classes = 20 + 1

optimizer.lr = 5e-3
schedulers.multistep.milestones = [70000, 9000]
train.epochs = 40

train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(RandomHorizontalFlip)(),
        L(Resize)(imshape="${train.imshape}"),
        L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    ]
)
val_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(ToTensor)(),
        L(Resize)(imshape="${train.imshape}"),
    ]
)
gpu_transform = L(torchvision.transforms.Compose)(
    transforms=[L(Normalize)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
data_train.dataset = L(torch.utils.data.ConcatDataset)(
    datasets=[
        L(VOCDataset)(
            data_dir=get_dataset_dir("f{VOCDEVKIT_DATASET_DIR}/VOC2007"),
            split="train",
            transform=train_cpu_transform,
            keep_difficult=True,
            remove_empty=True,
        ),
        L(VOCDataset)(
            data_dir=get_dataset_dir("f{VOCDEVKIT_DATASET_DIR}/VOC2012"),
            split="train",
            transform=train_cpu_transform,
            keep_difficult=True,
            remove_empty=True,
        ),
    ]
)
data_val.dataset = L(VOCDataset)(
    data_dir=get_dataset_dir("f{VOCDEVKIT_DATASET_DIR}/VOC2007"),
    split="val",
    transform=val_cpu_transform,
    remove_empty=False,
)
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(VOCDataset.class_names)}
