from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from neuralvision.helpers import batch_collate
from neuralvision.tops.config.instantiate import instantiate
from neuralvision.tops.config.lazy import LazyConfig
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from PIL import Image, ImageFont
from vizer.draw import draw_boxes

plt.rcParams["text.usetex"] = True
sns.set()


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


def load_annotation_file(cfg: OmegaConf, set_type: str) -> Union[Dict, None]:
    if set_type == "train":
        _file = cfg["data_train"]["dataset"]["annotation_file"]
    elif set_type == "val":
        _file = cfg["data_val"]["dataset"]["annotation_file"]
    else:
        raise ValueError(f"Unknown set_type: {set_type}")
    if _file.endswith(".json"):
        with open(_file, "r") as f:
            return json.load(f)
    else:
        raise NotImplementedError("Only json are supported!")


def get_avg_box_metrics(
    annotations: pd.DataFrame, unique_catagories: List[str]
) -> pd.DataFrame:
    table = {"dimension": ["area", "aspect", "width", "height"]}
    for category in unique_catagories:
        table[category] = _get_avg_box_metric(annotations, category)
    return pd.DataFrame(table)


def _get_avg_box_metric(
    annotations: pd.DataFrame, category_id: str
) -> Tuple[float, float, float, str]:

    """Returns the average width and height of the bounding boxes."""
    blob = annotations[annotations["category_id"] == category_id]
    blob = blob.drop(columns=["id", "image_id", "category_id"]).describe().round(2)
    mean_width = blob["width"]["mean"]
    mean_heigth = blob["height"]["mean"]
    mean_area = round(mean_width * mean_heigth, 2)
    mean_aspect = _calculate_aspect(mean_width, mean_heigth)
    return (mean_area, mean_aspect, blob["width"]["mean"], blob["height"]["mean"])


def _calculate_aspect(width: int, height: int) -> str:
    width, height = int(width), int(height)

    def gcd(a, b):
        """highest number that evenly divides both width and height."""
        return a if b == 0 else gcd(b, a % b)

    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)

    return f"{x}:{y}"


def view_aspect_ratio_distribution(cleaned_annotations: pd.DataFrame) -> None:
    """
    View the distribution of aspect ratios of the bounding boxes.
    """

    aspects = []

    for row in cleaned_annotations.iterrows():
        aspect = _calculate_aspect(row[1]["width"], row[1]["height"])
        a, b = aspect.split(":")
        aspects.append(int(a) / int(b))

    aspects = pd.Series(aspects)
    min_ratio = aspects.min().round(2)
    max_ratio = aspects.max().round(2)

    # Find outlier
    # outliers = aspects[np.abs(stats.zscore(aspects)) < 3]

    outliers = aspects[~((aspects - aspects.mean()).abs() > 3 * aspects.std())]

    q_low = aspects.quantile(0.01)
    q_hi = aspects.quantile(0.99)

    mean_aspect = aspects.mean().round(2)

    plt.figure(dpi=120)
    sns.histplot(aspects)
    plt.axvline(q_low, color="r", linestyle="--", label="1\\% quantile")
    plt.axvline(q_hi, color="r", linestyle="--", label="99\\% quantile")
    plt.axvline(mean_aspect, color="g", linestyle="--", label="mean")
    plt.title(
        "Aspect Ratio Distribution (min: {}, max: {})".format(min_ratio, max_ratio)
    )
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Number of boxes")
    plt.xlim(0, 6)
    plt.legend()
    plt.show()


def read_tfscalar_to_jsonfiles(scalar_path: Union[Path, str]) -> List[dict]:
    """reads a TF scaler.json file and converts it to a list of dictionaries"""
    with open(scalar_path, "r") as f:
        jsonfiles = f.readlines()
    # Convert to list of dictionaries
    return [json.loads(line) for line in jsonfiles]


def convert_scalar_to_metrics_map(scalar_path: Union[Path, str]) -> pd.DataFrame:
    jsonfiles = read_tfscalar_to_jsonfiles(scalar_path)
    # Find the dict containing anything mAP related (e.g. metrics/mAP )
    map_key = "metrics/mAP"
    metrics = [jsonfile for jsonfile in jsonfiles if map_key in jsonfile]
    return pd.DataFrame(metrics)


def convert_scalar_to_metrics_loss(scalar_path: Union[Path, str]) -> pd.DataFrame:
    jsonfiles = read_tfscalar_to_jsonfiles(scalar_path)
    # Find the dict containing anything loss related (e.g. metrics/mAP )
    map_key = "loss/total_loss"
    metrics = [jsonfile for jsonfile in jsonfiles if map_key in jsonfile]
    return pd.DataFrame(metrics)


def vizualize_metrics(
    metric_frame: pd.DataFrame, metric_types: List[str], axis_cfg: dict
) -> None:
    plt.figure(figsize=(6, 2), dpi=300)
    for metric_type in metric_types:
        sns.lineplot(
            x="global_step",
            y=metric_type,
            markers=True,
            dashes=False,
            data=metric_frame,
            label=metric_type.split("/")[1],
        )
    # Scale axis if needed
    if axis_cfg is not None:
        if axis_cfg["x"] is not None:
            plt.xlim(axis_cfg["x"][0], axis_cfg["x"][1])
        if axis_cfg["y"] is not None:
            plt.ylim(axis_cfg["y"][0], axis_cfg["y"][1])
    plt.xlabel("Global Steps")
    plt.ylabel("(Mean) Average Precision")
    plt.title("Metric: (m)AP")
    plt.legend()
    plt.show()


def vizualize_loss(
    loss_frame: pd.DataFrame, loss_types: List[str], axis_cfg: dict = None
) -> None:
    plt.figure(figsize=(6, 2), dpi=300)
    for loss_type in loss_types:
        sns.lineplot(
            x="global_step",
            y=loss_type,
            data=loss_frame,
            label=loss_type.split("/")[1],
        )
    # Scale axis if needed
    if axis_cfg is not None:
        if axis_cfg["x"] is not None:
            plt.xlim(axis_cfg["x"][0], axis_cfg["x"][1])
        if axis_cfg["y"] is not None:
            plt.ylim(axis_cfg["y"][0], axis_cfg["y"][1])
    plt.xlabel("Global Steps")
    plt.yticks
    plt.ylabel("Loss")
    plt.title("Metric: Loss")

    plt.legend()
    plt.show()


def preprocess_sample(sample, tfms, std, mean) -> Tuple[np.ndarray, np.ndarray]:

    # Reapply transforms
    sample = tfms(sample)

    # Normalize
    image = sample["image"] * std + mean
    image = (image * 255).byte()[0]
    boxes = sample["boxes"][0]

    # Percentage size to pixel size
    boxes[:, [0, 2]] *= image.shape[-1]
    boxes[:, [1, 3]] *= image.shape[-2]
    # Swap axes to match Matplotlib
    image = image.permute(1, 2, 0)
    # Detach
    image = image.cpu().numpy()
    boxes = boxes.cpu().numpy()
    return image, boxes


def visualize_sample(
    img_id,
    image,
    boxes,
    labels,
    lblmap: DictConfig,
    viz_cfg: dict = None,
):
    if viz_cfg is None:
        viz_cfg = {"figsize": (30, 20), "dpi": 120}

    # Add boxes
    im = draw_boxes(
        image=image,
        boxes=boxes,
        labels=labels,
        class_name_map=lblmap,
        width=2,
        font=ImageFont.truetype("Dyuthi-Regular.ttf", size=12),
    )

    plt.figure(figsize=viz_cfg["figsize"], dpi=viz_cfg["dpi"])
    plt.axis("off")
    plt.imshow(im)
    plt.title(f"Image ID: {img_id}", fontsize=12, fontweight="bold")
    plt.show()


def save_sample(
    img_id,
    image,
    boxes,
    labels,
    lblmap: DictConfig,
    viz_cfg: dict = None,
    save_dir: str = None,
):

    if save_dir is None:
        raise ValueError("save_dir must be specified!")

    if viz_cfg is None:
        if save_dir is not None:
            viz_cfg = {"figsize": (30, 20), "dpi": 300}
        else:
            viz_cfg = {"figsize": (30, 20), "dpi": 120}

    # Add boxes
    im = draw_boxes(
        image=image,
        boxes=boxes,
        labels=labels,
        class_name_map=lblmap,
        width=2,
        font=ImageFont.truetype("Dyuthi-Regular.ttf", size=12),
    )

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    im = Image.fromarray(im)
    im = im.resize((im.size[0] * 3, im.size[1] * 3), Image.ANTIALIAS)
    im.save(os.path.join(save_dir, f"{img_id}.jpg"))
