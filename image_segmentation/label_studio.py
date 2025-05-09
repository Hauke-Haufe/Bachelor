import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from label_studio_sdk import Client
import label_studio_converter.brush as brush
import matplotlib.pyplot as plt
import shutil
from PIL import Image


#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu

def create_plygon_tasks(path, run):
    """
    nutzt Sam um images aus einem run zu segmentieren um labelstudio task zu erstellen 
    
    Args:
        path (string): path to the zu den runs (data/dataset)
        run (string): den run der segmentiert werden soll
    
    """
    # === Parameters ===
    IOU_THRESHOLD = 0.2
    MIN_IOU_SCORE = 0.70
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
    files = [file for file in os.listdir(os.path.join(path, run)) if file.endswith(".png")]
    for file in files:
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

def create_masks(path, run):

    with open(path, "r")as file:
        coco_data = json.load(file)

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
    
    grouped = group_annotations_by_image_and_class(coco_data["annotations"])

    img_shapes = {img["id"]: (img["height"], img["width"]) for img in coco_data["images"]}
    catid_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    imgid_to_path = {img["id"]: Path(*Path(img["file_name"]).parts[-3:]) for img in coco_data["images"]}


    for (image_id, category_id), polygons in grouped.items():
        shape = img_shapes[image_id]
        mask = rasterize_polygon_segments(polygons, shape)
        path =  imgid_to_path[image_id]
        if catid_to_name[category_id] == "heu":
            full_path = os.path.join("data", "data_set", run , "unrefined", os.path.splitext(path.parts[-1])[0] +"heu.npy")
            np.save(full_path, mask)
        elif catid_to_name[category_id] == "cow":
            full_path = os.path.join("data", "data_set", run, "unrefined", os.path.splitext(path.parts[-1])[0] +"cow.npy")
            np.save(full_path, mask)
    
def import_masks_task(project, run):
    
    classes = ["cow", "heu"]

    #am besten api key nicht auf github
    ls = Client(url="http://localhost:8080", api_key="179e598bf40ebfc8904c1987e2507c0fe11936f0")
    project = ls.get_project(project) 

    def mask2labelstudioresult(task_id, rle, label):
        return {
                "from_name": "tag",   # BrushLabels name
                "to_name": "image",   # Image field name
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [label]
                }
            }
    

    # Load tasks
    project_tasks = project.get_tasks()

    # Mask folder
    mask_path = os.path.join("data","data_set", run, "unrefined")


    for task in tqdm(project_tasks):

        results = []
        for c in classes:
            mask_filename = os.path.splitext(task['data']['image'])[0] + f"{c}.npy"
            mask_filename =Path(mask_filename).parts[-1]
            
            if os.path.exists(os.path.join(mask_path, mask_filename)):
                mask = np.load(os.path.join(mask_path, mask_filename))

                # Binarize mask
                mask = (mask > 0).astype(np.uint8) * 255

                # Convert to Label Studio RLE
                rle = brush.mask2rle(mask)

                # Upload the prediction
                results.append(mask2labelstudioresult(task['id'], rle, c))
        
        project.create_prediction(
            task_id=task['id'],
            model_version=None,
            result= results
        )

def make_final_mask(refined_mask_folder, image_path, json_path, output_folder):

    def combine_masks(dim , mask_cow, mask_heu):

        
        if mask_cow is None:
            mask_cow = np.zeros(dim, dtype = np.uint8) 
        else:
            mask_cow = mask_cow[0]
        if mask_heu is None:
            mask_heu = np.zeros(dim, dtype = np.uint8)
        else:
            mask_heu = mask_heu[0]

        mask_cow = np.resize(mask_cow, dim)
        mask_heu = np.resize(mask_heu, dim)
        combined = np.zeros(dim, dtype = np.uint8)

        label1 = 1
        label2 = 2

        combined[mask_heu == 255] = label2 
        combined[mask_cow == 255] = label1
        

        plt.imshow(combined)
        #plt.show()

        return combined

    with open(json_path, "r") as f:
        labels = json.load(f)
    
    files = [file for file in os.listdir(refined_mask_folder)]

    for label in labels:
        pred = label["annotations"][0]["result"]
        dim = (pred[0]["original_height"],pred[0]["original_width"] )
        id  = label["id"]
        masks = [file for file in files if file.split("-")[1] == str(id)]
        cow_masks = [np.load(os.path.join(refined_mask_folder , file)) for file in masks if file.split("-")[-2] == "cow"]
        heu_masks = [np.load(os.path.join(refined_mask_folder , file))  for file in masks if file.split("-")[-2] == "heu"]

        if not len(cow_masks) >= 1:
            cow_masks = None
        if not len(heu_masks) >= 1:
            heu_masks = None

        combined_mask = combine_masks(dim, cow_masks, heu_masks)
        filename = Path(label["data"]["image"].split("=")[-1]).parts[-1]
        Image.fromarray(combined_mask).save(os.path.join(os.path.abspath(output_folder), "masks",  filename))
        #np.save(os.path.join(os.path.abspath(output_folder), "masks",  filename.strip(".png")), combined_mask)
        shutil.copy(os.path.join(image_path, filename), os.path.join(output_folder, "images", filename))
        



if __name__ == "__main__":
    
    #create_masks("data/data_set/run5/result.json", "run5")
    #import_masks_task(9, "run5")
    #create_plygon_tasks("data/data_set", "run5")
    make_final_mask("data/data_set/run5/refined","data/data_set/run5", "data/data_set/run5/project-9-at-2025-05-06-11-46-0d6a4d02.json", "dataset/run5")