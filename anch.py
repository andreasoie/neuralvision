from core.tops.config.lazy import LazyConfig
from core.tops.config.instantiate import instantiate

cfg = LazyConfig.load("core/configs/task24/retina_P1_7.py")
anchors = instantiate(cfg.anchors)(order="xywh")
print("Number of anchors: ", len(anchors), " should be 65520")
