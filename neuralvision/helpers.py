from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from neuralvision.tops.config.lazy import LazyConfig


def load_config(config_path: PathLike):
    config_path = Path(config_path)
    print(f"Loading config from: {config_path}")
    run_name = "_".join(config_path.parts[1:-1]) + "_" + config_path.stem
    cfg = LazyConfig.load(str(config_path))
    cfg.output_dir = Path(cfg.train._output_dir).joinpath(
        *config_path.parts[1:-1], config_path.stem
    )
    cfg.run_name = run_name
    print("--" * 40)
    return cfg


def batch_collate(batch):
    elem = batch[0]
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem}
    return batch_


def batch_collate_val(batch):
    """
    Same as batch_collate, but removes boxes/labels from dataloader
    """
    elem = batch[0]
    ignore_keys = set(("boxes", "labels"))
    batch_ = {
        key: default_collate([d[key] for d in batch])
        for key in elem
        if key not in ignore_keys
    }
    return batch_


def class_id_to_name(labels, label_map: list):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    return [label_map[idx] for idx in labels]


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]


def bbox_ltrb_to_ltwh(boxes_ltrb: Union[np.ndarray, torch.Tensor]):
    cat = torch.cat if isinstance(boxes_ltrb, torch.Tensor) else np.concatenate
    assert boxes_ltrb.shape[-1] == 4
    return cat((boxes_ltrb[..., :2], boxes_ltrb[..., 2:] - boxes_ltrb[..., :2]), -1)


def bbox_center_to_ltrb(boxes_center: Union[np.ndarray, torch.Tensor]):
    cat = torch.stack if isinstance(boxes_center, torch.Tensor) else np.stack
    assert boxes_center.shape[-1] == 4
    cx, cy, w, h = [boxes_center[..., i] for i in range(4)]
    return cat(
        (
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ),
        -1,
    )


def bbox_ltrb_to_center(boxes_lrtb: Union[np.ndarray, torch.Tensor]):
    cat = torch.stack if isinstance(boxes_lrtb, torch.Tensor) else np.stack
    assert boxes_lrtb.shape[-1] == 4
    l, t, r, b = [boxes_lrtb[..., i] for i in range(4)]
    return cat((0.5 * (l + r), 0.5 * (t + b), r - l, b - t), -1)
