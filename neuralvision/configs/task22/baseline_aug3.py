# Inherit configs from the default ssd300
import torchvision

from neuralvision.datasets_classes.tdt4265_dataset import TDT4265Dataset
from neuralvision.tops.config.lazy import LazyCall as L
from neuralvision.transforms.gpu_transforms import ColorJitter, Normalize
from neuralvision.transforms.target_transform import GroundTruthBoxesToAnchors
from neuralvision.transforms.transform import RandomSampleCrop, Resize, ToTensor

from neuralvision.configs.dir_utils import get_dataset_dir

# absolute import causes issues, using relative imports
from ..template.ssd300 import (
    train,
    anchors,
    optimizer,
    schedulers,
    backbone,
    model,
    data_train,
    data_val,
    loss_objective,
)

TDT4265_DATASET_DIR = "datasets/tdt4265"

# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)
train.image_channels = 3
train.epochs = 100
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
train_gpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(ColorJitter)(brightness=0.1, contrast=0.1, saturation=0.1),
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878]),
    ]
)
val_gpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ]
)
data_train["dataset"] = L(TDT4265Dataset)(
    img_folder=get_dataset_dir(TDT4265_DATASET_DIR),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir(f"{TDT4265_DATASET_DIR}/train_annotations.json"),
)
data_val["dataset"] = L(TDT4265Dataset)(
    img_folder=get_dataset_dir(TDT4265_DATASET_DIR),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir(f"{TDT4265_DATASET_DIR}/val_annotations.json"),
)
data_train["gpu_transform"] = train_gpu_transform
data_val["gpu_transform"] = val_gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
