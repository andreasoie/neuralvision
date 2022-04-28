from collections import OrderedDict

import numpy as np
import torch
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from scripts.save_comparison_images import convert_boxes_coords_to_pixel_coords
from vizer.draw import draw_boxes


def tensor_ordered_dict(boxes, categories, scores):
    return OrderedDict((("boxes", boxes), ("labels", categories), ("scores", scores)))


class BackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x) -> OrderedDict:
        boxes, categories, scores = self.model(x)[0]
        return tensor_ordered_dict(boxes, categories, scores)


def pro_predict(input_tensor, batch, model, detection_threshold):
    boxes, categories, scores = model(
        input_tensor, score_threshold=detection_threshold
    )[0]
    boxes = convert_boxes_coords_to_pixel_coords(
        boxes.detach().cpu(), batch["width"], batch["height"]
    )
    categories = categories.cpu().numpy().tolist()
    return boxes, categories, scores


def retinafpn_reshape_transform(features: tuple):
    target_size = (1, 8)  # select one bigger maybe later
    activations = []
    for feat in features:
        activations.append(
            torch.nn.functional.interpolate(torch.abs(feat), target_size)
        )
    activations = torch.cat(activations, axis=1)
    return activations


def renormalize_cam_in_bounding_boxes(
    bounding_boxes, canvas_image, grey_cam_img, boxes, labels, scores, class_name_map
):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes."""
    renormalized_cam = np.zeros(grey_cam_img.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in bounding_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grey_cam_img[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    renormalized_image = show_cam_on_image(canvas_image, renormalized_cam, use_rgb=True)
    # To remove catagory name and score probability from the vizualization,
    # you have to go into the vizer src code and set display_string to None
    image_with_bounding_boxes = draw_boxes(
        renormalized_image, boxes, labels, scores, class_name_map=class_name_map
    )
    return image_with_bounding_boxes
