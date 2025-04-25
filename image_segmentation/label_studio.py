import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import json
import os
from pycocotools import mask as mask_utils
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from label_studio_converter.brush import encode_rle, image2annotation
import uuid


#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu


def create_plygon_tasks(path):

    # === Parameters ===
    IOU_THRESHOLD = 0.2
    MIN_IOU_SCORE = 0.9
    MIN_STABILITY = 0.92
    MIN_AREA = 300

    def iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0

    def merge_masks(masks, iou_threshold):
        """
        Kombiniert Masken, die überlappen

        Args:
            masks (dict): segmentation Resultate aus dem SAM Netzwerk
            iou_threshold (float): Schwellenwert ab dem Segmente als übelappend gelten

        Returns:
            (list): kombinierte Masken
        """

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

    def mask_to_polygons(mask, min_area=100, min_points=3):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []

        for contour in contours:
            contour = contour.squeeze()

            # Ensure valid shape and point count
            if len(contour.shape) != 2 or len(contour) < min_points:
                continue

            # Filter by area (using cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            polygon = contour.flatten().tolist()
            polygons.append(polygon)

        return polygons
    
    sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    label_studio_annotations= []
    for run in os.listdir(path):
        for file in os.listdir(os.path.join(path, run)):
            image = cv2.imread(os.path.join(path, run, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = mask_generator.generate(image)

            filtered = [m for m in masks if
                m['area'] >= MIN_AREA and
                m['predicted_iou'] >= MIN_IOU_SCORE and
                m['stability_score'] >= MIN_STABILITY
            ]

            merged_masks = merge_masks(filtered, IOU_THRESHOLD)
            file_path = os.path.join("data_set", run, file)

            label_studio_annotations.append(
                {
                    "data": {
                        "image": f"/data/local-files/?d={file_path}"
                    },
                    "annotations": [{
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "image",
                                "type": "polygonlabels",
                                "value": {
                                    "points": [[x / image.shape[1] * 100, y / image.shape[0] * 100] for x, y in np.array(polygon).reshape(-1, 2)],
                                    "polygonlabels": ["object"]
                                }
                            }
                            for mask_data in merged_masks
                            for polygon in mask_to_polygons(mask_data) 
                        ]
                    }]
                }
            )
            
        with open(os.path.join(path, run, "polygon_task.json"), "w") as f:
            json.dump(label_studio_annotations, f, indent=4)


def coco_from_polygons_to_masks(coco_data, path):

    def group_annotations_by_image_and_class(annotations):
        """
        Groups COCO annotations by (image_id, category_id)
        Returns: dict[(image_id, category_id)] = list of polygons
        """
        grouped = defaultdict(list)
        for ann in annotations:
            key = (ann['image_id'], ann['category_id'])
            for seg in ann['segmentation']:
                grouped[key].append(seg)

        return grouped

    def rasterize_polygon_segments(polygons, image_shape):

        mask = np.zeros(image_shape, dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], color=1)

        return mask
    
    def binary_mask_to_labelstudio_rle_safe(mask):
        mask = mask.transpose()
        mask = np.array((np.array(mask) > 128) * 255, dtype=np.uint8)
        array = mask.ravel()
        array = np.repeat(array, 4)
        rle = encode_rle(array)
        return rle

    def build_mask_task(image_path, annotation):

        return {
            "data": {
                "image": f"/data/local-files/?d={image_path}"
            },
            "predictions": annotation
        }
    
    def build_annotation(label_name, mask, shape):
        
        rle_counts =   binary_mask_to_labelstudio_rle_safe(mask)
        return{
            "result": [
                        {
                            "id": str(uuid.uuid4())[0:8],
                            "type": "brushlabels",
                            "value": {"rle": rle_counts, "format": "rle", "brushlabels": [label_name]},
                            "origin": "manual",
                            "to_name": "tag",
                            "from_name": "image",
                            "image_rotation": 0,
                            "original_width": shape[1],
                            "original_height": shape[0],
                        }
                    ],
        }
    
    def test(mask):

        cv2.imwrite("temp.png", mask)

        annotation = image2annotation(
        "temp.png",
        label_name='Airplane',
        from_name='tag',
        to_name='image',
        model_version='v1',
        score=0.5,
    )

        # prepare Label Studio Task
        task = {
            'data': {'image': 'https://labelstud.io/images/test.jpg'},
            'predictions': [annotation],
        }

        """ You can import this `task.json` to the Label Studio project with this labeling config:

        <View>
        <Image name="image" value="$image" zoom="true"/>
        <BrushLabels name="tag" toName="image">
            <Label value="Airplane" background="rgba(255, 0, 0, 0.7)"/>
            <Label value="Car" background="rgba(0, 0, 255, 0.7)"/>
        </BrushLabels>
        </View>

        """
        json.dump(task, open('task.json', 'w'))

    grouped = group_annotations_by_image_and_class(coco_data["annotations"])

    img_shapes = {img["id"]: (img["height"], img["width"]) for img in coco_data["images"]}
    catid_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    imgid_to_path = {img["id"]: Path(*Path(img["file_name"]).parts[-3:]) for img in coco_data["images"]}

    segmentations = {img['id']: [] for img in coco_data['images']}
    for (image_id, category_id), polygons in grouped.items():
        shape = img_shapes[image_id]
        mask = rasterize_polygon_segments(polygons, shape)
        #segmentations[image_id].append(build_annotation(catid_to_name[category_id], mask, shape))
        test(mask)

    tasks = []
    for (image_id), segmentation in segmentations.items():
        
        task = build_mask_task(imgid_to_path[image_id], segmentation)
        tasks.append(task)

    with open(os.path.join(path, "mask_task.json"), "w") as f:
        json.dump(tasks, f, indent=2)


#segment_images("data\data_set")

with open("data/result.json", ) as f:
    coco_data = json.load(f)

coco_from_polygons_to_masks(coco_data, "data/data_set/run1")

