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
from pathlib import Path
import datetime



#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu

#sync up mit Labelstudio images
#implement sanity checks with the label studio database for the run


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

def add_storage():
        # Connect to Label Studio
    ls = Client(url='http://localhost:8080', api_key='<your_api_token>')

    # Define storage settings
    storage_config = {
        'path': '/absolute/path/to/your/images',  # Must be accessible to the server
        'use_blob_urls': True,                    # Allows file:// URI usage
        'title': 'My Local Storage',
        'description': 'Local dataset for image labeling',
        'regex_filter': r'.*\.(jpg|png|jpeg)$',   # Optional: limit file types
        'use_file_name_as_label': False
    }

    # Create local file import storage
    response = ls.make_request(
        'post',
        '/api/storages/localfiles',
        json=storage_config
    )

def mask2labelstudioresult(rle, label):
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

#class for labeling and managing working with Labelstudio
class label_project:

    def __init__(self, root_path = "data/data_set"):
        
        self.root = Path(root_path)
        with open(self.root / "progress.json") as f:
            progress = json.load(f)

        self.backup_path = self.root / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)

        self.classes = ["heu", "cow"]
        self.prio_rule = [1, 0]

        self.runs = progress["runs"]
        self.key = progress["api_key"]
        self.host = progress["host"]
        self.polygon_config = progress["polygon_config"]
        self.brush_config = progress["brush_config"]
    
    #--------------------------------------
    #   Projectmanger functions
    #--------------------------------------
    #adds a run to the project with the images without any labelstudio project beeing created
    def add_run(self, image_folder: str):
        
        image_folder = Path(image_folder)

        found = False
        key = 1
        while not found:
            if str(key) in self.runs:
                key +=1
            else:
                found = True
        
        run_path =(self.root / f"run{key}")
        run_path.mkdir(parents=True, exist_ok=True)
        
        self.runs[str(key)] = {"polygon_project": None, 
                                  "brush_project": None, 
                                  "images": [str(run_path/ file.name) for file in image_folder.iterdir() if file.suffix == ".png"]
                                  }
        
        for file in (image_folder).iterdir():
            if file.suffix == ".png":
                shutil.copy(file, str(run_path/ file.name))
        
        self.save_config()
    
    #danger zone deletes a run from the progress json and the images get deleted
    def delete_run(self, run: int):

        if os.path.exists(self.root / f"run{run}"):
            shutil.rmtree(self.root / f"run{run}")
        self.runs.pop(str(run), None)
        self.save_config()

    #saves the current config of the project to the progress json
    def save_config(self):

        with open(self.root / "progress.json", "w") as f:
            json.dump({"runs": self.runs,
                        "api_key": self.key,
                        "host": self.host,
                        "polygon_config": self.polygon_config,
                        "brush_config":  self.brush_config}, f, indent=4)

    #saves the actual progress for all runs in from of the labelstudio jsons to the backup folder
    def backup_progress(self):

        for run, _ in self.runs.items():
            self.backup_run(run)

    #saves the actual progress for one run in form of the labelstudio jsons to the backup folder
    def backup_run(self, run, type = "polygon"):

        ls = Client(url=self.host, api_key=self.key)
        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task to save only polygon and brush available")

        if self.runs[run][f"{type}_project"] is not None:
            project = ls.get_project(self.runs[run][f"{type}_project"]) 
            project_tasks = project.get_tasks()

            backup_run_path = self.backup_path / f"run{run}_{type}.json"
            if backup_run_path.is_file():
                with open(backup_run_path, "r") as f:
                    backup = json.load(f)

                with open(backup_run_path, "w") as f:
                    backup.append({"date":str(datetime.datetime.now()),"data": project_tasks})
                    json.dump(backup, f)
            else:
                with open(backup_run_path, "w") as f:
                    json.dump([{"date":str(datetime.datetime.now()),"data": project_tasks}], f)

            print(f"Backup von {type}_task run {run} erstellt")
        
    #--------------------------------------
    #         old label workflow
    #--------------------------------------
    #the old Labeling pipline consist out of two part. First the Segment anything Model (Sam) is used
    #to segment any images and create a polygon labeling project in Label Studio. This is supposed to get
    #coarse lables to start with. The polygon need to be sorted and labeled. 
    #In the next step the labled polygons get converted to a brush task in Lablestudio. This is for the mask 
    #refinement and creation of the finals Masks

    #creates a new Labelstudio project and uses Automaticsegmenter from Sam to create polygon segmentations
    def create_polygon_task(self, run: int):

        run = str(run)
        if not run in self.runs:
            raise RuntimeError("the run does not exists") 

        ls = Client(url=self.host, api_key=self.key)

        if self.runs[run]["polygon_project"] is None:
            project = ls.start_project(
                title = f"Polygon_tasks{run}",
                label_config = self.polygon_config
            )

            self.runs[run]["polygon_project"] = project.id
            self.save_config()
        else:
            print("WARNING: a polygon_project already exits the progress would be overwritten. Type yes to proceed")
            if input() == "yes":

                self.backup_run(run, "polygon")
                project = ls.get_project(self.runs[run]["polygon_project"])
                project.delete_all_tasks()
            else:
                return

        
        # === Parameters ===
        IOU_THRESHOLD = 0.2
        MIN_IOU_SCORE = 0.70
        MIN_STABILITY = 0.92
        MIN_AREA = 300

        sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
        sam.to("cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)

        tasks= []
        files = self.runs[run]["images"]
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = mask_generator.generate(image)

            filtered = [m for m in masks if
                m['area'] >= MIN_AREA and
                m['predicted_iou'] >= MIN_IOU_SCORE and
                m['stability_score'] >= MIN_STABILITY
            ]

            merged_masks = merge_masks(filtered, IOU_THRESHOLD)
            file_path =  Path(*Path(file).parts[1:])

            tasks.append(
                {
                    "data": {
                        "image": f"/data/local-files/?d={file_path}"
                    },
                    "predictions": [{
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
            
        project.import_tasks(tasks)

    #uses the Polygon segmentation with actual lables to convert to a brush Labelstudio Project
    def create_brush_task(self, run: int):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if self.runs[run]["polygon_project"] is None:
            raise RuntimeError("u need a polygon task first for this run")
        
        if self.runs[run]["brush_project"] is None:
            brush_project = ls.start_project(
                title = f"Brush_tasks{run}",
                label_config = self.brush_config
            )

            self.runs[run]["brush_project"] = brush_project.id
            self.save_config()
        else:
            print("WARNING: u already have existing progress in this runs brush task. If u want to overide it type yes")
            if input() == "yes":
                self.backup_run(run, "brush")
                brush_project = ls.get_project(self.runs[run]["brush_project"]) 
                brush_project.delete_all_tasks()
            else:
                return
        
        project = ls.get_project(self.runs[run]["polygon_project"]) 
        project_tasks = project.get_tasks()

        for i in range(len(project_tasks)): 
            task = project_tasks[i]

            results = []
            class_polygon_dict = {}
            for label in self.classes:
                class_polygon_dict[label] = []
            
            for polygon in task["annotations"][0]["result"]:
                points = polygon["value"]["points"]
                label = polygon["value"]["polygonlabels"][0]

                if label in self.classes:
                    class_polygon_dict[label].append(points)

            height = task["annotations"][0]["result"][0]["original_height"]
            width = task["annotations"][0]["result"][0]["original_width"]

            for label in self.classes:
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in class_polygon_dict[label]:
                    poly = [[int(point[0]/ 100 * width +1), int(point[1]/ 100 *height +1)] for point in poly]
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], color=1)

                plt.imshow(mask)
                plt.show()
                mask = (mask > 0).astype(np.uint8) * 255
                rle = brush.mask2rle(mask)
                results.append(mask2labelstudioresult(rle, label))
            

            brush_project.import_tasks({
                "data": {
                    "image": task["data"]["image"]
                    }})
            
            brush_task = brush_project.get_tasks()[i]
            brush_project.create_prediction(
                task_id=brush_task['id'],
                model_version=None,
                result= results
            )
    
    #exports all brush labled masks
    def export_final_masks(self, run: int, destination_folder):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if self.runs[run]["brush_project"] is not None:
            project = ls.get_project(self.runs[run]["brush_project"]) 
        else:
            raise RuntimeError("u need to have a brush project before exporting it")
        
        tasks = project.get_tasks()

        for task in tqdm(tasks):
            
            if len(task["annotations"]) > 0:
                if len(task["annotations"][0]["result"]) >0:
                    mask_by_clas_dict = {}
                    height = task["annotations"][0]["result"][0]["original_height"]
                    width = task["annotations"][0]["result"][0]["original_width"]

                    for label in self.classes:
                        mask_by_clas_dict[label] = np.zeros((height, width))

                    for result in task["annotations"][0]["result"]:
                        mask = brush.decode_rle(result["value"]["rle"]).reshape((height, width, 4))[:, :, 3]
                        
                        
                        label = result["value"]["brushlabels"][0]
                        mask_by_clas_dict[label] = np.logical_or(mask, mask_by_clas_dict[label])

        #still need to be overlapt
                    

def save_json(root):

    root = Path(root)
    runs = {"runs": {1: {"polygon_project": None, "brush_project": None, "images": [str(file) for file in (root / "run1").iterdir() if str(file).endswith(".png")] },
                2: {"polygon_project": None, "brush_project": None,  "images": [str(file) for file in (root / "run2").iterdir()if str(file).endswith(".png")]},
                3: {"polygon_project": 7, "brush_project": 13,  "images": [str(file) for file in (root / "run3").iterdir()if str(file).endswith(".png")]},
                4: {"polygon_project": None, "brush_project": None,  "images":[str(file) for file in (root / "run4").iterdir()if str(file).endswith(".png")]},
                5: {"polygon_project": 8, "brush_project": 9, "images":[str(file) for file in (root / "run5").iterdir()if str(file).endswith(".png")]},
                6: {"polygon_project": 11, "brush_project": 14,  "images": [str(file) for file in (root / "run6").iterdir()if str(file).endswith(".png")]}
                },
            "api_key": "179e598bf40ebfc8904c1987e2507c0fe11936f0",
            "host": "http://localhost:8080",
            "polygon_config": """<View>
                <Header value="Select label and click the image to start"/>
                <Image name="image" value="$image" zoom="true"/>
                <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
                <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff00ff"/></PolygonLabels>
                </View>""",
            "brush_config" : """<View>
                <Image name="image" value="$image" zoom="true"/>
                <BrushLabels name="tag" toName="image">
                    
                    
                <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff0080"/><Label value="object" background="#FFC069"/></BrushLabels>
                </View>"""
            }
    
    with open(root /"progress.json", "w") as f:
        json.dump(runs, f)


def make_final_mask(refined_mask_folder, image_path, json_path, output_folder):

    #label studio task format

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
        plt.show()

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
    
    #save_json("data/data_set")
    project = label_project()
    project.export_final_masks(7, "data")

    #create_plygon_tasks("data/data_set", "run6")
    #create_masks("data/data_set/run3/result.json", "run3")
    #import_masks_task(14, "run6")
    #make_final_mask("data/data_set/run3/refined","data/data_set/run3", "data/data_set/run3/project-13-at-2025-05-27-17-04-c8737056.json", "dataset/runs/run3")

    