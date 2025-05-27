import cv2
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

i=0

for path in Path("data/data_set/run3").iterdir():
    print(path.is_file())
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run prediction
    outputs = predictor(image)

    if i % 10 == 0:
        # Visualize masks
        v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Show output
        plt.imshow(out.get_image())
        plt.axis("off")
        plt.show()

        # Access pixelwise masks directly
        masks = outputs["instances"].pred_masks.cpu().numpy()  # Shape: [N, H, W]
        classes = outputs["instances"].pred_classes.cpu().numpy()

        # Save the first mask as an example
        if masks.shape[0] > 0:
            first_mask = masks[0].astype(np.uint8) * 255
            cv2.imwrite("mask0.png", first_mask)
    i += 1