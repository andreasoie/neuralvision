import pathlib

import click
import cv2
import numpy as np
import torch
import tqdm
import os

from neuralvision.helpers import load_config
from neuralvision.tops.checkpointer.checkpointer import load_checkpoint
from neuralvision.tops.config.instantiate import instantiate
from neuralvision.tops.torch_utils import get_device, to_cuda
from vizer.draw import draw_boxes


VIDEO_DIR_SRC = pathlib.Path("datasets/tdt4265/videos")
VIDEO_DIR_DST = pathlib.Path("outputs/demo/videos")

assert VIDEO_DIR_SRC.exists()
assert VIDEO_DIR_DST.exists()


@torch.no_grad()
@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.argument("video_name", type=click.Path(dir_okay=True, path_type=str))
@click.argument("save_name", type=click.Path(dir_okay=True, path_type=str))
@click.option(
    "-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=0.5
)
def run_demo(config_path: str, score_threshold: float, video_name: str, save_name: str):
    cfg = load_config(config_path)
    model = to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(
        cfg.output_dir.joinpath("checkpoints"), map_location=get_device()
    )
    model.load_state_dict(ckpt["model"])

    # Setup save format
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    width, height = 1024, 128
    FPS = 30

    # Setup read source
    # /home/andy/dev/neuralvision/datasets/tdt4265/videos/Video00010_combined.avi
    read_as = str(VIDEO_DIR_SRC.joinpath(video_name))
    reader = cv2.VideoCapture(read_as)

    # Setup write destination
    save_as = str(VIDEO_DIR_DST.joinpath(save_name))
    writer = cv2.VideoWriter(save_as, fourcc, FPS, (width, height))

    # Recursively instantiate objects
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    video_length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    assert reader.isOpened()
    for frame_idx in tqdm.trange(video_length, desc="Predicting on video"):
        ret, frame = reader.read()
        assert ret, "An error occurred"
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        img = cpu_transform({"image": frame})["image"].unsqueeze(0)
        img = to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        boxes, categories, scores = model(img, score_threshold=score_threshold)[0]
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes, categories, scores = [
            _.cpu().numpy() for _ in [boxes, categories, scores]
        ]
        frame = draw_boxes(frame, boxes, categories, scores).astype(np.uint8)
        writer.write(frame[:, :, ::-1])
    print("Video saved to:", pathlib.Path(save_as).absolute())


if __name__ == "__main__":
    run_demo()
