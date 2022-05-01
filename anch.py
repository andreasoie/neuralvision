from core.tops.config.lazy import LazyConfig
from core.tops.config.instantiate import instantiate
from matplotlib import pyplot as plt


cfg = LazyConfig.load("core/configs/task24/retina_P1_7.py")
anchors = instantiate(cfg.anchors)(order="xywh")
print("Number of anchors: ", len(anchors), " should be 65520")

raw_anchr = instantiate(cfg.anchors)
print(raw_anchr)
boxes = cfg.anchors["feature_map_boxes"]

first_boxes = boxes[0]

print("First boxes: ", first_boxes)
