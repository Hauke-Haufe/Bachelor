import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pycocotools import mask as mask_utils
import json


#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu

# === Parameters ===
IOU_THRESHOLD = 0.7
MIN_IOU_SCORE = 0.9
MIN_STABILITY = 0.92

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def merge_masks(masks, iou_threshold):
    merged = []
    used = [False] * len(masks)

    for i, m1 in enumerate(masks):
        if used[i]:
            continue
        combined_mask = m1['segmentation'].copy()
        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            m2 = masks[j]
            if iou(combined_mask, m2['segmentation']) > iou_threshold:
                combined_mask = np.logical_or(combined_mask, m2['segmentation'])
                used[j] = True
        used[i] = True
        merged.append(combined_mask)
    return merged


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) != 2 or len(contour) < 3:
            continue  # Ignore invalid polygons
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons

sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
#sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("/home/haufe/Schreibtisch/Uni/beachlor/Beachlor/data/istockphoto-1282514444-612x612.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

filtered = [m for m in masks if
    m['predicted_iou'] >= MIN_IOU_SCORE and
    m['stability_score'] >= MIN_STABILITY
]

merged_masks = merge_masks(filtered, IOU_THRESHOLD)

annotations = []
for i, mask_data in enumerate(merged_masks):
    mask = mask_data['segmentation']
    polygons = mask_to_polygons(mask)

    for polygon in polygons:
        annotation = {
            "id": i,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [polygon],  
            "area": float(mask_data["area"]),
            "bbox": mask_data["bbox"],
            "iscrowd": 0
        }
        annotations.append(annotation)

label_studio_annotations = [{
    "data": {
        "image": "/data/local-files/?d=dataset1/istockphoto-1282514444-612x612.jpg"
    },
    "annotations": [{
        "result": [{
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": [[x / image.shape[1] * 100, y / image.shape[0] * 100] for x, y in np.array(polygon).reshape(-1, 2)],
                "polygonlabels": ["object"]
            }
        } for polygon in mask_to_polygons(mask_data['segmentation'])]
    }]
} for i, mask_data in enumerate(masks)]

with open("annotions.json", "w") as f:
    json.dump(label_studio_annotations, f, indent=4)