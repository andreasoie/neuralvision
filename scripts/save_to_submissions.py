import json
import pathlib
import click
import torch
import tqdm

from neuralvision.helpers import bbox_ltrb_to_ltwh, load_config
from neuralvision.tops.checkpointer.checkpointer import load_checkpoint
from neuralvision.tops.config.instantiate import instantiate
from neuralvision.tops.torch_utils import AMP, get_device, to_cuda


@torch.no_grad()
@click.command()
@click.argument("config_path")
@click.argument("save_path", type=pathlib.Path)
def get_detections(config_path, save_path):
    cfg = load_config(config_path)
    model = instantiate(cfg.model)
    model.eval()
    ckpt = load_checkpoint(
        cfg.output_dir.joinpath("checkpoints"), map_location=get_device()
    )
    model.load_state_dict(ckpt["model"])
    model = to_cuda(model)
    data_val = instantiate(cfg.data_val.dataloader)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    detections = []

    for batch in tqdm.tqdm(data_val, desc="Evaluating on dataset"):
        batch["image"] = to_cuda(batch["image"])
        batch = gpu_transform(batch)
        with torch.cuda.amp.autocast(enabled=AMP()):
            # You can change the nms IOU threshold!
            predictions = model(
                batch["image"],
                nms_iou_threshold=0.50,
                max_output=200,
                score_threshold=0.05,
            )

        for idx in range(len(predictions)):
            boxes_ltrb, categories, scores = predictions[idx]
            # ease-of-use for specific predictions
            H, W = batch["height"][idx], batch["width"][idx]
            box_ltwh = bbox_ltrb_to_ltwh(boxes_ltrb)
            box_ltwh[:, [0, 2]] *= W
            box_ltwh[:, [1, 3]] *= H
            box_ltwh, category, score = [
                x.cpu() for x in [box_ltwh, categories, scores]
            ]
            img_id = batch["image_id"][idx].item()
            for b_ltwh, label_, prob_ in zip(box_ltwh, category, score):
                detections.append(
                    dict(
                        image_id=img_id,
                        category_id=int(label_),
                        score=prob_.item(),
                        bbox=b_ltwh.tolist(),
                    )
                )

    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "w") as fp:
        json.dump(detections, fp)

    print("Detections saved to: ", save_path)
    print("Absolute path: ", save_path.absolute())
    URL_SUBISSIONS = "https://tdt4265-annotering.idi.ntnu.no/submissions/"
    print(f"Go to: {URL_SUBISSIONS} to submit your result")


if __name__ == "__main__":
    get_detections()
