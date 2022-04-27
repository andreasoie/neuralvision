import sys

assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
from pathlib import Path

import click
import torch
import tqdm
from torch.optim.lr_scheduler import ChainedScheduler
import argparse

from core.evaluate import evaluate
from core.helpers import load_config, tencent_trick
from core.tops import build, torch_utils
from core.tops.checkpointer import checkpointer
from core.tops.config.instantiate import instantiate
from core.tops.logger import logger
from core.tops.misc import print_module_summary, print_config

torch.backends.cudnn.benchmark = True

arg = argparse.ArgumentParser()
arg.add_argument("--cfg", type=str, default="ssd300.py")
ags = arg.parse_args()


def main():
    evaluate_only = False
    path = Path(ags.cfg)
    cfg = load_config(path)
    # print_config(cfg)
    print(ags.cfg, sep=" ")

    build.init(cfg.output_dir)
    torch_utils.set_AMP(cfg.train.amp)
    torch_utils.set_seed(cfg.train.seed)

    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)

    cocoGt = dataloader_val.dataset.get_annotations_as_coco()

    model = torch_utils.to_cuda(instantiate(cfg.model))

    print(sum(p.numel() for p in model.parameters()))

    optimizer = instantiate(cfg.optimizer, params=tencent_trick(model))

    scheduler = ChainedScheduler(
        instantiate(list(cfg.schedulers.values()), optimizer=optimizer)
    )

    checkpointer.register_models(
        dict(model=model, optimizer=optimizer, scheduler=scheduler)
    )

    total_time = 0
    if checkpointer.has_checkpoint():
        train_state = checkpointer.load_registered_models(load_best=False)
        total_time = train_state["total_time"]
        logger.log(
            f"Resuming train from: epoch: {logger.epoch()}, \
                global step: {logger.global_step()}"
        )

    gpu_transform_val = instantiate(cfg.data_val.gpu_transform)
    gpu_transform_train = instantiate(cfg.data_train.gpu_transform)

    evaluation_fn = functools.partial(
        evaluate,
        model=model,
        dataloader=dataloader_val,
        cocoGt=cocoGt,
        gpu_transform=gpu_transform_val,
        label_map=cfg.label_map,
    )

    if evaluate_only:
        evaluation_fn()
        exit()

    # Helps perform the steps of gradient scaling conveniently
    scaler = torch.cuda.amp.GradScaler(enabled=torch_utils.AMP())

    dummy_input = torch_utils.to_cuda(
        torch.randn(2, cfg.train.image_channels, *cfg.train.imshape)
    )

    print_module_summary(model, (dummy_input,))


if __name__ == "__main__":
    SystemExit(main())
