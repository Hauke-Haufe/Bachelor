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
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def recall(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    recall = inter / mask2.sum() if mask2.sum() > 0 else 0.0
    return recall

def automatic():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
        
    i=0

    for path in Path("data/data_set/run5").iterdir():

        print(path.is_file())
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run prediction
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        labels = instances.pred_classes.numpy()
        masks = outputs["instances"].pred_masks.cpu().numpy()

        for j, box in enumerate(boxes):
            x0, y0, x1, y1 = map(int, box)
            cropped_img = image[y0:y1, x0:x1]
            
            sam_masks = mask_generator.generate(cropped_img)

            global_masks = []
            for mask in sam_masks:
                full_mask = np.zeros(image.shape[0:2])
                full_mask[y0:y1, x0:x1] = mask["segmentation"]

                if recall(masks[j], full_mask)> 0.3:
                    global_masks.append(full_mask)
                    plt.imshow(image)
                    plt.imshow(masks[j], alpha=0.5)
                    plt.imshow(full_mask, alpha=0.5)
                    plt.show()
                    
        i += 1

def sample_mask_point(mask):
    print(mask.shape)
    found = False
    while not found:
        x = np.random.randint(0,mask.shape[0])
        y = np.random.randint(0,mask.shape[1])
        
        if mask[x,y] == 1:
            found = True
    
    return np.array([x, y])

def prompt():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
    sam.to("cuda")
    sam = SamPredictor(sam)
        
    i=0
    for path in Path("data/data_set/run5").iterdir():

        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run prediction
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        labels = instances.pred_classes.numpy()
        masks = outputs["instances"].pred_masks.cpu().numpy()

        sam.set_image(image)
        for j, box in enumerate(boxes):
            
            x0, y0, x1, y1 = map(int, box)
            cropped_img = image[y0:y1, x0:x1]

            points = []
            labels = []
            for k in range(5):

                point = sample_mask_point(masks[j])
                points.append(point)
                labels.append(1)

            
            sam_masks, scores, logits = predictor.predict(
                point_coords=np.asanyarray(points),
                point_labels=np.asanyarray(labels),
                multimask_output=True,
            )

            global_masks = []
            for mask in sam_masks:
                full_mask = np.zeros(image.shape[0:2])
                full_mask[y0:y1, x0:x1] = mask["segmentation"]

                if recall(masks[j], full_mask)> 0.3:
                    global_masks.append(full_mask)
                    plt.imshow(image)
                    plt.imshow(masks[j], alpha=0.5)
                    plt.imshow(full_mask, alpha=0.5)
                    plt.show()
                    


        """if i % 5 == 0:
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
                cv2.imwrite("mask0.png", first_mask)"""
        i += 1

prompt()