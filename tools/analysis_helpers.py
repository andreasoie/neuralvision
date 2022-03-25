import matplotlib.pyplot as plt
import torch
import torchvision

plt.rcParams["text.usetex"] = True

from neuralvision.helpers import batch_collate
from neuralvision.tops.config.instantiate import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from vizer.draw import draw_boxes


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


def yield_and_vizulize_sample(
    it: _MultiProcessingDataLoaderIter,
    lblmap: DictConfig,
    tfms: torchvision.transforms.Compose,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    viz_cfg: dict,
):
    """supported keys for single sample:
    Key [image]: shape=torch.Size([1, 3, 128, 1024]), dtype=torch.float32
    Key [boxes]: shape=torch.Size([1, 14, 4]), dtype=torch.float32
    Key [labels]: shape=torch.Size([1, 14]), dtype=torch.int64
    Key [width]: shape=torch.Size([1]), dtype=torch.int64
    Key [height]: shape=torch.Size([1]), dtype=torch.int64
    Key [image_id]: shape=torch.Size([1]), dtype=torch.int64
    """

    # Preprocess
    sample = next(it)
    sample = tfms(sample)
    image = sample["image"] * image_std + image_mean
    image = (image * 255).byte()[0]
    boxes = sample["boxes"][0]
    boxes[:, [0, 2]] *= image.shape[-1]
    boxes[:, [1, 3]] *= image.shape[-2]

    # Add boxes
    im = image.permute(1, 2, 0).cpu().numpy()
    im = draw_boxes(
        image=im,
        boxes=boxes.cpu().numpy(),
        labels=sample["labels"][0].cpu().numpy().tolist(),
        class_name_map=lblmap,
        width=2,
    )

    plt.figure(figsize=viz_cfg["figsize"], dpi=viz_cfg["dpi"])
    plt.axis("off")
    plt.imshow(im)
    plt.show()
