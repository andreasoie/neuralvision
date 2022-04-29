# Inherit configs from the default ssd300
import numpy as np
import torchvision
from core.backbones.resnetfpn import ResnetFPN
from core.configs.dir_utils import get_dataset_dir
from core.datasets_classes.tdt4265_dataset import TDT4265Dataset
from core.ssd.retinanet import RetinaNet
from core.tops.config.lazy import LazyCall as L
from core.transforms.gpu_transforms import (
    ColorJitter,
    Normalize,
    RandomAdjustSharpness,
)
from core.transforms.target_transform import GroundTruthBoxesToAnchors
from core.transforms.transform import (
    RandomHorizontalFlip,
    RandomSampleCrop,
    Resize,
    ToTensor,
)

from core.ssd.custom_boxes import CustomBoxes

# absolute import causes issues, using relative imports
from ..template.ssd300 import (
    anchors,
    backbone,
    data_train,
    data_val,
    loss_objective,
    model,
    optimizer,
    schedulers,
    train,
)

TDT4265_DATASET_DIR = "datasets/tdt4265"

# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)  # type: ignore
train.image_channels = 3  # type: ignore
train.epochs = 50
train.batch_size = 8
# anchors.aspect_ratios = [ #
#     [2.5, 3.5],
#     [2.5, 3.5],
#     [2.5, 3.5],
#     [2.5, 3.5],
#     [2.5, 3.5],
#     [2.5, 3.5],
# ]

anchors = L(CustomBoxes)(
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
    #     type	car	person	scooter	bicycle	rider	bus	truck
    # 0	area	582.27	194.11	278.82	284.76	238.62	494.21	894.82
    # 1	aspect	22:25	7:26	9:29	13:20	2:7	19:25	24:35
    # 2	width	22.56	7.42	9.41	13.73	8.51	19.26	24.96
    # 3	height	25.81	26.16	29.63	20.74	28.04	25.66	35.85
    # ------------------------------------------------------------------
    # rider:    0.15 to 0.5
    # person:   0.15 to 0.45
    # bicycle:  0.3 to 1
    # scooter:  0.1 to 0.5
    # car:      0.5 to 1.5
    # truck:    0.4 to 1.4 except middle
    # bus:      0.4 to 1
    aspect_map={
        0: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
        1: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
        2: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
        3: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
        4: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
        5: [5.0, 3.7037037, 2.56410256, 1.72413793, 1.19047619, 0.76335878],
    },
    # aspect_map={
    #     0: np.divide(1, np.linspace(0.15, 0.5, 6)).round(4).tolist(),
    #     1: np.divide(1, np.linspace(0.15, 0.45, 6)).round(4).tolist(),
    #     2: np.divide(1, np.linspace(0.3, 1, 6)).round(4).tolist(),
    #     3: np.divide(1, np.linspace(0.1, 0.5, 6)).round(4).tolist(),
    #     4: np.divide(1, np.linspace(0.5, 1.5, 6)).round(4).tolist(),
    #     5: np.divide(1, np.linspace(0.4, 1, 6)).round(4).tolist(),
    # },
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2,
)

NUM_CLASSES = 8 + 1  # Add 1 for background

backbone = L(ResnetFPN)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=NUM_CLASSES,
    use_deeper_head=False,
    use_weightstyle=False,
)

train_cpu_transform = L(torchvision.transforms.Compose)(
    transforms=[
        L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(Resize)(imshape="${train.imshape}"),
        L(RandomHorizontalFlip)(),
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
        L(RandomAdjustSharpness)(sharpness_factor=1.5),
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
data_val["gpu_transform"] = val_gpu_transform
data_train["gpu_transform"] = train_gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
