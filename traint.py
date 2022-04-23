import sys

assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
from pathlib import Path

import click
import torch
import tqdm
from torch.optim.lr_scheduler import ChainedScheduler

from neuralvision.evaluate import evaluate
from neuralvision.helpers import load_config, tencent_trick
from neuralvision.tops import build, torch_utils
from neuralvision.tops.checkpointer import checkpointer
from neuralvision.tops.config.instantiate import instantiate
from neuralvision.tops.logger import logger
from neuralvision.tops.misc import print_module_summary, print_config

torch.backends.cudnn.benchmark = True


def train_epoch(
    model,
    scaler: torch.cuda.amp.GradScaler,
    optim,
    dataloader_train,
    scheduler,
    gpu_transform: torch.nn.Module,
    log_interval: int,
):
    grad_scale = scaler.get_scale()
    for batch in tqdm.tqdm(dataloader_train, f"Epoch {logger.epoch()}"):
        batch = torch_utils.to_cuda(batch)
        batch["labels"] = batch["labels"].long()
        batch = gpu_transform(batch)

        for k, v in batch.items():
            print(k, v.shape)

        with torch.cuda.amp.autocast(enabled=torch_utils.AMP()):
            output = model(batch["image"], batch["boxes"])
        scaler.scale(output["loss"]).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        if grad_scale == scaler.get_scale():
            scheduler.step()
            if logger.global_step() % log_interval:
                logger.add_scalar(
                    "stats/learning_rate", scheduler._schedulers[-1].get_last_lr()[-1]
                )
        else:
            grad_scale = scaler.get_scale()
            logger.add_scalar("amp/grad_scale", scaler.get_scale())
        if logger.global_step() % log_interval == 0:
            to_log = {f"loss/{k}": v.mean().cpu().item() for k, v in output.items()}
            logger.add_dict(to_log)
        # torch.cuda.amp skips gradient steps if backward pass produces NaNs/infs.
        # If it happens in the first iteration, scheduler.step() will throw exception
        logger.step()
    return


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--evaluate-only", default=False, is_flag=True, help="Eval only")
def train(config_path: Path, evaluate_only: bool):

    logger.DEFAULT_SCALAR_LEVEL = logger.DEBUG

    cfg = load_config(config_path)
    print_config(cfg)

    build.init(cfg.output_dir)
    torch_utils.set_AMP(cfg.train.amp)
    torch_utils.set_seed(cfg.train.seed)

    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)

    # cocoGt = dataloader_val.dataset.get_annotations_as_coco()

    model = cfg.model["model"]

    optimizer = instantiate(cfg.optimizer, params=tencent_trick(model))

    scheduler = ChainedScheduler(
        instantiate(list(cfg.schedulers.values()), optimizer=optimizer)
    )

    checkpointer.register_models(
        dict(model=model, optimizer=optimizer, scheduler=scheduler)
    )

    total_time = 0
    # if checkpointer.has_checkpoint():
    #     train_state = checkpointer.load_registered_models(load_best=False)
    #     total_time = train_state["total_time"]
    #     logger.log(
    #         f"Resuming train from: epoch: {logger.epoch()}, \
    #             global step: {logger.global_step()}"
    #     )

    # gpu_transform_val = instantiate(cfg.data_val.gpu_transform)
    gpu_transform_train = instantiate(cfg.data_train.gpu_transform)

    # evaluation_fn = functools.partial(
    #     evaluate,
    #     model=model,
    #     dataloader=dataloader_val,
    #     cocoGt=cocoGt,
    #     gpu_transform=gpu_transform_val,
    #     label_map=cfg.label_map,
    # )

    # if evaluate_only:
    #     evaluation_fn()
    #     exit()

    # Helps perform the steps of gradient scaling conveniently
    scaler = torch.cuda.amp.GradScaler(enabled=torch_utils.AMP())

    # dummy_input = torch_utils.to_cuda(
    #     torch.randn(1, cfg.train.image_channels, *cfg.train.imshape)
    # )

    # print_module_summary(model.model, (dummy_input,))

    start_epoch = logger.epoch()
    for _ in range(start_epoch, cfg.train.epochs):
        start_epoch_time = time.time()
        train_epoch(
            model,
            scaler,
            optimizer,
            dataloader_train,
            scheduler,
            gpu_transform_train,
            cfg.train.log_interval,
        )
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        logger.add_scalar("stats/epoch_time", end_epoch_time)

        # eval_stats = evaluation_fn()
        # eval_stats = {f"metrics/{key}": val for key, val in eval_stats.items()}
        # logger.add_dict(eval_stats, level=logger.INFO)
        # train_state = dict(total_time=total_time)
        # checkpointer.save_registered_models(train_state)
        logger.step_epoch()
    logger.add_scalar("stats/total_time", total_time)


if __name__ == "__main__":
    train()
