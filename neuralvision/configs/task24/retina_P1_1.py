# Inherit configs from the default ssd300
import torchvision
from neuralvision.backbones.resnetfpn import ResnetFPN
from neuralvision.configs.dir_utils import get_dataset_dir
from neuralvision.datasets_classes.tdt4265_dataset import TDT4265Dataset
from neuralvision.ssd.retinanet import RetinaNet
from neuralvision.tops.config.lazy import LazyCall as L
from neuralvision.transforms.gpu_transforms import (
    ColorJitter,
    Normalize,
    RandomAdjustSharpness,
)
from neuralvision.transforms.target_transform import GroundTruthBoxesToAnchors
from neuralvision.transforms.transform import (
    RandomHorizontalFlip,
    RandomSampleCrop,
    Resize,
    ToTensor,
)

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
train.epochs = 50
train.batch_size = 8


# [0.02, 0.03, 0.04, 0.06, 0.1, 0.19]
# anchors.aspect_ratios = [
#     [0.1, 0.1],
#     [0.1, 0.1],
#     [0.1, 0.1],
#     [0.1, 0.1],
#     [0.1, 0.1],
#     [0.2, 0.2],
# ]

"""
	dimension	car	person	scooter	bicycle	rider	bus	truck
0	area	740.36	190.92	278.52	282.77	242.52	1761.49	3075.34
1	aspect	14:13	7:26	9:29	13:20	2:7	45:38	28:27
2	width	28.41	7.14	9.4	13.7	8.45	45.54	56.15
3	height	26.06	26.74	29.63	20.64	28.7	38.68	54.77

"""
anchors.aspect_ratios = [
    [7 / 26, 7 / 26],
    [9 / 29, 9 / 29],
    [2 / 7, 2 / 7],
    [13 / 20, 13 / 20],
    [14 / 13, 14 / 13],
    [15 / 14, 15 / 14],
]

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
