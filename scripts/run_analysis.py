import time
from pathlib import Path

import click
import torch

from core.tops.torch_utils import get_device, to_cuda
from core.tops.checkpointer.checkpointer import load_checkpoint
from core.tops.config.instantiate import instantiate
from core.helpers import load_config


@torch.no_grad()
def evaluation(cfg, N_images: int, do_sysout: bool = True):
    model = instantiate(cfg.model)
    model.eval()
    model = to_cuda(model)
    ckpt = load_checkpoint(
        cfg.output_dir.joinpath("checkpoints"),
        map_location=get_device(),
        do_sysout=do_sysout,
    )
    model.load_state_dict(ckpt["model"])
    dataloader_val = instantiate(cfg.data_val.dataloader)
    batch = next(iter(dataloader_val))
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    batch = to_cuda(batch)
    batch = gpu_transform(batch)
    images = batch["image"]
    imshape = list(images.shape[2:])
    # warmup
    for i in range(10):
        model(images)
    start_time = time.time()
    for i in range(N_images):
        _ = model(images)
    total_time = time.time() - start_time

    _name = cfg["run_name"]
    print(
        f" \n {_name} | {N_images / total_time} images/sec | {total_time} runtime (s)"
    )


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("-n", "--n-images", default=500, type=int)
def main(config_path: Path, n_images: int):
    do_sysout = False
    cfg = load_config(config_path, do_sysout)
    evaluation(cfg, n_images, do_sysout)


if __name__ == "__main__":
    main()
