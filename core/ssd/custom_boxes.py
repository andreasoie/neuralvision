# Modified from: https://github.com/lufficc/SSD
import numpy as np
import torch
from typing import Dict, List
from math import sqrt


# class CustomBoxes(object):
#     def __init__(
#         self,
#         image_shape: tuple,
#         feature_sizes: List[tuple],
#         min_sizes: List[int],
#         strides: List[tuple],
#         aspect_ratios: List[int],
#         scale_center_variance: float,
#         scale_size_variance: float,
#     ):
#         """Generate SSD anchors Boxes.
#         It returns the center, height and width of the anchors.
#         The values are relative to the image size
#         Args:
#             image_shape: tuple of (image height, width)
#             feature_sizes: each tuple in the list is the feature shape
#                         outputted by the backbone (H, W)
#         Returns:
#             anchors (num_priors, 4): The prior boxes represented as
#                         [[center_x, center_y, w, h]].
#                         All the values are relative to the image size.
#         """
#         self.scale_center_variance = scale_center_variance
#         self.scale_size_variance = scale_size_variance
#         self.num_boxes_per_fmap = [2 + 2 * len(ratio) for ratio in aspect_ratios]
#         # Calculation method slightly different from paper

#         """
#         """
#         self.feature_map_boxes = {}

#         anchors = []
#         # size of feature and number of feature
#         for fidx, [fH, fW] in enumerate(feature_sizes):
#             bbox_sizes = []
#             h_min = min_sizes[fidx][0] / image_shape[0]  #
#             w_min = min_sizes[fidx][1] / image_shape[1]
#             bbox_sizes.append((w_min, h_min))
#             h_max = sqrt(min_sizes[fidx][0] * min_sizes[fidx + 1][0]) / image_shape[0]
#             w_max = sqrt(min_sizes[fidx][1] * min_sizes[fidx + 1][1]) / image_shape[1]
#             bbox_sizes.append((w_max, h_max))
#             for r in aspect_ratios[fidx]:
#                 if 1 < fidx < 4:
#                     bbox_sizes.append((w_min / sqrt(r), h_min * sqrt(r)))
#                     bbox_sizes.append((w_min / sqrt(r * 1.2), h_min * sqrt(r * 1.2)))
#                 elif 4 <= fidx < 6:
#                     bbox_sizes.append((w_min * sqrt(r), h_min / sqrt(r)))
#                     bbox_sizes.append((w_min * sqrt(r * 1.2), h_min / sqrt(r * 1.2)))
#                 else:
#                     bbox_sizes.append((w_min / sqrt(r), h_min * sqrt(r)))
#                     bbox_sizes.append((w_min * sqrt(r), h_min / sqrt(r)))

#             scale_y = image_shape[0] / strides[fidx][0]
#             scale_x = image_shape[1] / strides[fidx][1]
#             self.feature_map_boxes[f"{fH}x{fW}"] = bbox_sizes
#             for w, h in bbox_sizes:
#                 for i in range(fH):
#                     for j in range(fW):
#                         cx = (j + 0.5) / scale_x
#                         cy = (i + 0.5) / scale_y
#                         anchors.append((cx, cy, w, h))

#         self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()

#         self.anchors_ltrb = self.anchors_xywh.clone()
#         self.anchors_ltrb[:, 0] = (
#             self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
#         )
#         self.anchors_ltrb[:, 1] = (
#             self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
#         )
#         self.anchors_ltrb[:, 2] = (
#             self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
#         )
#         self.anchors_ltrb[:, 3] = (
#             self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]
#         )

#     def __call__(self, order):
#         if order == "ltrb":
#             return self.anchors_ltrb
#         if order == "xywh":
#             return self.anchors_xywh

#     @property
#     def scale_xy(self):
#         return self.scale_center_variance

#     @property
#     def scale_wh(self):
#         return self.scale_size_variance


class CustomBoxes(object):
    def __init__(
        self,
        image_shape: tuple,
        feature_sizes: List[tuple],
        min_sizes: List[int],
        strides: List[tuple],
        aspect_map: Dict[int, List[int]],
        scale_center_variance: float,
        scale_size_variance: float,
    ):
        """Generate SSD anchors Boxes.
        It returns the center, height and width of the anchors.
        The values are relative to the image size
        Args:
            image_shape: tuple of (image height, width)
            feature_sizes: each tuple in the list is the feature shape
                        outputted by the backbone (H, W)
        Returns:
            anchors (num_priors, 4): The prior boxes represented as
                        [[center_x, center_y, w, h]].
                        All the values are relative to the image size.
        """
        self.aspect_map = aspect_map
        self.scale_center_variance = scale_center_variance
        self.scale_size_variance = scale_size_variance
        self.num_boxes_per_fmap = [len(ratios) for ratios in self.aspect_map.values()]

        anchors = []
        self.feature_map_boxes = {}

        for fidx, [fH, fW] in enumerate(feature_sizes):
            h_min = min_sizes[fidx][0] / image_shape[0]  #
            w_min = min_sizes[fidx][1] / image_shape[1]
            bbox_sizes = []
            for r in self.aspect_map[fidx]:
                bbox_sizes.append((w_min * sqrt(r), h_min / sqrt(r)))
            scale_y = image_shape[0] / strides[fidx][0]
            scale_x = image_shape[1] / strides[fidx][1]
            self.feature_map_boxes[f"{fH}x{fW}"] = bbox_sizes

            for w, h in bbox_sizes:
                # print(
                #     f"[{fidx}] featureMap [{fH}, {fW}] - size ({round(w*image_shape[1], 2)}, {round(h*image_shape[0], 2)})"
                # )
                for i in range(fH):
                    for j in range(fW):
                        cx = (j + 0.5) / scale_x
                        cy = (i + 0.5) / scale_y
                        anchors.append((cx, cy, w, h))

        self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()
        self.anchors_ltrb = self.anchors_xywh.clone()
        self.anchors_ltrb[:, 0] = (
            self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
        )
        self.anchors_ltrb[:, 1] = (
            self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
        )
        self.anchors_ltrb[:, 2] = (
            self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
        )
        self.anchors_ltrb[:, 3] = (
            self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]
        )

    def __call__(self, order):
        if order == "ltrb":
            return self.anchors_ltrb
        if order == "xywh":
            return self.anchors_xywh

    @property
    def scale_xy(self):
        return self.scale_center_variance

    @property
    def scale_wh(self):
        return self.scale_size_variance
