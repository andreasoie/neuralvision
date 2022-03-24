# Inherit configs from the default ssd300
import os
import torchvision

from neuralvision.datasets_classes.tdt4265_dataset import TDT4265Dataset
from neuralvision.tops.config.lazy import LazyConfig as L
from neuralvision.transforms.gpu_transforms import Normalize
from neuralvision.transforms.target_transform import GroundTruthBoxesToAnchors
from neuralvision.transforms.transform import Resize, ToTensor

from dir_utils import get_dataset_dir
from ssd300 import data_train, data_val, model, train

TDT4265_DATASET_DIR = "datasets/tdt4265"

# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)
train.image_channels = 3
model.num_classes = 8 + 1  # Add 1 for background class


train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(ToTensor)(),
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
    transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ]
)
data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir(TDT4265_DATASET_DIR),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir(f"{TDT4265_DATASET_DIR}/train_annotations.json"),
)
data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir(TDT4265_DATASET_DIR),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir(f"{TDT4265_DATASET_DIR}/val_annotations.json"),
)
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
