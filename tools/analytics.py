from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import json
from typing import Dict, List, Tuple, Union

plt.rcParams["text.usetex"] = True

from neuralvision.tops.config.lazy import LazyConfig
from neuralvision.helpers import batch_collate
from neuralvision.tops.config.instantiate import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from vizer.draw import draw_boxes
from PIL import ImageFont


def get_config(config_path: str, batch_size: int = 1):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = batch_size
    return cfg


def prepare_dataset(cfg: OmegaConf, dataset_type: str, batch_size: int):
    cfg.train.batch_size = batch_size

    if dataset_type == "train":
        # Remove GroundTruthBoxesToAnchors transform
        if cfg.data_train.dataset._target_ == torch.utils.data.ConcatDataset:
            for dataset in cfg.data_train.dataset.datasets:
                dataset.transform.transforms = dataset.transform.transforms[:-1]
        else:
            cfg.data_train.dataset.transform.transforms = (
                cfg.data_train.dataset.transform.transforms[:-1]
            )
        dataset = instantiate(cfg.data_train.dataloader)
        gpu_transform = instantiate(cfg.data_train.gpu_transform)
    else:
        cfg.data_val.dataloader.collate_fn = batch_collate
        dataset = instantiate(cfg.data_val.dataloader)
        gpu_transform = instantiate(cfg.data_val.gpu_transform)

    return cfg, dataset, gpu_transform


def calculate_mean_std(cfg: OmegaConf) -> Tuple[torch.tensor, torch.tensor]:
    """Assumes that the first GPU transform is Normalize
    - If it fails, just change the index from 0."""
    image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(
        1, 3, 1, 1
    )
    image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(
        1, 3, 1, 1
    )
    return image_mean, image_std


def get_category_metrics(category_ids: pd.DataFrame, decimals: float = 2) -> None:
    # Series to DataFrame
    label_distribution = pd.DataFrame(
        category_ids.value_counts().sort_values(ascending=False)
    )
    # Rename
    label_distribution.index.name = "category"
    # Rename
    label_distribution.rename(columns={"category_id": "total"}, inplace=True)
    # Cummulative sum
    total_labels = label_distribution["total"].sum()
    # Add percentage column for simpler intuitive analysis
    label_distribution["in %"] = label_distribution["total"] / total_labels
    # Prettify
    label_distribution["in %"] = label_distribution["in %"] * 100
    label_distribution["in %"] = label_distribution["in %"].round(decimals)
    label_distribution.reset_index(inplace=True)
    return label_distribution


def process_annotations(loaded_annotations: List[dict]) -> pd.DataFrame:
    cleaned_annotations = []
    for annotation in loaded_annotations:
        # Ignore segmentation
        del annotation["segmentation"]
        # Rescaled image sizes
        xmin = annotation["bbox"][0]
        ymin = annotation["bbox"][1]
        xmax = annotation["bbox"][2] + xmin
        ymax = annotation["bbox"][3] + ymin
        # Create rows with rescaled coordinates for
        # easier visualization and analysis
        cleaned_annotations.append(
            {
                "id": annotation["id"],
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "area": annotation["area"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": xmax - xmin,
                "height": ymax - ymin,
            }
        )
    return pd.DataFrame(cleaned_annotations)


def process_categories(loaded_catagories: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(loaded_catagories).drop(columns=["supercategory"])


def load_annotation_file(cfg: OmegaConf) -> Union[Dict, None]:
    _file = cfg["data_train"]["dataset"]["annotation_file"]
    if _file.endswith(".json"):
        with open(_file, "r") as f:
            return json.load(f)
    else:
        raise NotImplementedError("Only json are supported!")


def visualize_sample(
    it: _MultiProcessingDataLoaderIter,
    lblmap: DictConfig,
    tfms: torchvision.transforms.Compose,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    viz_cfg: dict = None,
):
    """supported keys for single sample:
    Key [image]: shape=torch.Size([1, 3, 128, 1024]), dtype=torch.float32
    Key [boxes]: shape=torch.Size([1, 14, 4]), dtype=torch.float32
    Key [labels]: shape=torch.Size([1, 14]), dtype=torch.int64
    Key [width]: shape=torch.Size([1]), dtype=torch.int64
    Key [height]: shape=torch.Size([1]), dtype=torch.int64
    Key [image_id]: shape=torch.Size([1]), dtype=torch.int64
    """
    if viz_cfg is None:
        viz_cfg = {"figsize": (30, 20), "dpi": 120}

    sample = next(it)
    sample = tfms(sample)

    # Preprocess
    image = sample["image"] * image_std + image_mean
    image = (image * 255).byte()[0]
    boxes = sample["boxes"][0]
    boxes[:, [0, 2]] *= image.shape[-1]
    boxes[:, [1, 3]] *= image.shape[-2]

    # Add boxes
    im = image.permute(1, 2, 0).cpu().numpy()
    custom_font = ImageFont.truetype("Dyuthi-Regular.ttf", size=12)
    im = draw_boxes(
        image=im,
        boxes=boxes.cpu().numpy(),
        labels=sample["labels"][0].cpu().numpy().tolist(),
        class_name_map=lblmap,
        width=2,
        font=custom_font,
    )

    plt.figure(figsize=viz_cfg["figsize"], dpi=viz_cfg["dpi"])
    plt.axis("off")
    plt.imshow(im)
    plt.title(f"Image ID: {sample['image_id'][0]}", fontsize=12, fontweight="bold")
    plt.show()
