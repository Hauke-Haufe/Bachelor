import json
from pathlib import Path
import os
from tqdm import tqdm
import shutil
import datetime

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from label_studio_sdk import Client
import label_studio_converter.brush as brush

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from LabelwithSam import LabelwithSam, LWSContext

#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu

#sync up mit Labelstudio images

#wichitg filepath darf nicht mit der labaelstudio root anfangen sonder erst dannach

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


#class for labeling and managing working with Labelstudio
class label_project:

    def __init__(self, root_path = "data/data_set"):
        
        self.root = Path(root_path)
        if (self.root / "progress.json").is_file():
            with open(self.root / "progress.json") as f:
                progress = json.load(f)
        else:
            print("No progress detected.")
            print("Enter the Labelstudio Api key:")
            api_key = input()
            print("Enter the Url on which labelstudio runs: ")
            host = input()
            progress = {"runs": {}, "api_key":api_key, "host": host, 
                        "polygon_config":
                        """<View>
                        <Header value="Select label and click the image to start"/>
                        <Image name="image" value="$image" zoom="true"/>
                        <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
                        <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff00ff"/></PolygonLabels>
                        </View>""",
                        "brush_config": """<View>
                        <Image name="image" value="$image" zoom="true"/>
                        <BrushLabels name="tag" toName="image">
                        <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff0080"/><Label value="object" background="#FFC069"/></BrushLabels>
                        </View>""",
                        }
            
        self.runs = progress["runs"]
        self.key = progress["api_key"]
        self.host = progress["host"]
        self.configs = {"brush": progress["brush_config"], "polygon": progress["polygon_config"]}

        self.save_config()
        o= os.environ.copy()
        try:
            self.label_studio_root = Path(os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'])
        except KeyError:
            raise RuntimeError("No local dataset is set for Label Studio. Set the LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT Enviorment Variable")
        try:
            os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED']
        except KeyError:
            raise RuntimeError("Local File serving isnt enabled. Set the LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED Enviorment Variable")

        self.backup_path = self.root / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)

        self.classes = {"heu": {"label":2, "prio": 0}, "cow": {"label":1, "prio": 1}}

    #--------------------------------------
    #   util functions
    #--------------------------------------

    def lpath_to_path(self, lpath):
        p =  self.label_studio_root /Path(lpath.split("=")[-1])
        return p
    
    @staticmethod   
    def masktolabelstudioresult(mask, label):
            rle = brush.mask2rle(mask)
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

    def add_project(self, run, type, i_import = True):

        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task only polygon and brush available")
        
        if self.project_exits(run, type):
            raise RuntimeError(f"a {type} project already exits")

        ls = Client(url=self.host, api_key=self.key)
        project = ls.start_project(
                title = f"{type}_tasks{run}",
                label_config = self.configs[type]
            )

        self.runs[run][f"{type}_project"] = project.id
        self.save_config()

        if i_import:
            for image in self.runs[run]["images"]:
                path = Path(*Path(image).parts[1:])
                project.import_tasks({
                "data": {
                    "image": f"/data/local-files/?d={str(path)}"
                    }})
            
    #danger zone deletes a run from the progress json and the images get deleted
    def delete_run(self, run: int):

        if os.path.exists(self.root / f"run{run}"):
            shutil.rmtree(self.root / f"run{run}")
        self.runs.pop(str(run), None)
        self.save_config()

    def delete_all_tasks(self, run, type):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task to delete only polygon and brush available")

        if self.runs[run] is not None:
            self.backup_run(run, type)
            brush_project = ls.get_project(self.runs[run][f"{type}_project"]) 
            brush_project.delete_all_tasks()
        else:
            print("No such Project exists")

    def delete_project(self, run, type):
        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task to delete only polygon and brush available")

        if self.runs[run][f"{type}_project"] is not None:

            project = ls.get_project(self.runs[run][f"{type}_project"])
            self.backup_run(run, type)
            project.delete()
            self.runs[run][f"{type}_project"] = None
            self.save_config()

        else:
            raise RuntimeError(f"{type} project doesent exists for run {run}")

    #nicht getestet
    def reset_tasks(self, run, type):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task only polygon and brush available")

        if self.runs[run][f"{type}_project"] is not None:
            project = ls.get_project(self.runs[run][f"{type}_project"])
            self.backup_run(run, type)
            project.delete_all_tasks()

            for image in self.runs[run]["images"]:
                path = Path(*Path(image).parts[1:])
                project.import_tasks({
                "data": {
                    "image": f"/data/local-files/?d={str(path)}"
                    }})
        else:
            raise RuntimeError(f"{type} project doesent exists for run {run}")

    #saves the current config of the project to the progress json
    def save_config(self):

        with open(self.root / "progress.json", "w") as f:
            json.dump({"runs": self.runs,
                        "api_key": self.key,
                        "host": self.host,
                        "polygon_config": self.configs["polygon"],
                        "brush_config": self.configs["brush"]}, f, indent=4)

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
    
    #nicht getestet
    def project_exits(self, run, type):

        if not(type == 'polygon' or type == 'brush'):
            raise RuntimeError("not a valid task only polygon and brush available")

        run = str(run)
        if not run in self.runs:
            raise RuntimeError("the run does not exists") 

        if self.runs[run][f"{type}_project"] is None:
            return False
        
        return True
    
    
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
        ls = Client(url=self.host, api_key=self.key)

        if not self.project_exits(run, "polygon"):
            self.add_project(run, "polygon", False)

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

        if not self.project_exits(run, "polygon"):
            raise RuntimeError("u need a polygon task first for this run")
        
        if not self.project_exits(run, "brush"):
            self.add_project(run, "brush", False)
        else:
            print("WARNING: u already have existing progress in this runs brush task. If u want to overide it type yes")
            if input() == "yes":
                self.reset_tasks(run, "brush")
            else:
                return
        
        project = ls.get_project(self.runs[run]["polygon_project"])
        brush_project = ls.get_project(self.runs[run]["brush_project"]) 
        project_tasks = project.get_tasks()

        for i in range(len(project_tasks)): 
            task = project_tasks[i]

            results = []
            class_polygon_dict = {}
            for label, _ in self.classes.items():
                class_polygon_dict[label] = []
            
            for polygon in task["annotations"][0]["result"]:
                points = polygon["value"]["points"]
                label = polygon["value"]["polygonlabels"][0]

                if label in self.classes:
                    class_polygon_dict[label].append(points)

            height = task["annotations"][0]["result"][0]["original_height"]
            width = task["annotations"][0]["result"][0]["original_width"]

            for label, _ in self.classes.items():
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in class_polygon_dict[label]:
                    poly = [[int(point[0]/ 100 * width +1), int(point[1]/ 100 *height +1)] for point in poly]
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], color=1)

                
                mask = (mask > 0).astype(np.uint8) * 255
                
                results.append(self.masktolabelstudioresult(mask, label))
            

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
    #assumes only one Annotation was made
    def export_final_masks(self, run: int, destination_folder):

        if os.path.exists( destination_folder):
            if os.path.isfile( destination_folder) or os.path.islink( destination_folder):
                os.remove( destination_folder)
            elif os.path.isdir( destination_folder):
                shutil.rmtree( destination_folder)
        
        
        Path(destination_folder).mkdir()
        (Path(destination_folder) / "images").mkdir()
        (Path(destination_folder) / "masks").mkdir()

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if self.project_exits(run, "brush"):
            project = ls.get_project(self.runs[run]["brush_project"]) 
        else:
            raise RuntimeError("u need to have a brush project before exporting it")
        
        tasks = project.get_tasks()
        labels = list(self.classes.keys())
        sorted_labels = sorted(labels, key=lambda l: self.classes[l]["prio"], reverse = False)

        for task in tqdm(tasks):
            
            if len(task["annotations"]) > 0:
                if len(task["annotations"][0]["result"]) >0:
                    mask_by_clas_dict = {}
                    height = task["annotations"][0]["result"][0]["original_height"]
                    width = task["annotations"][0]["result"][0]["original_width"]

                    for label, _ in self.classes.items():
                        mask_by_clas_dict[label] = np.zeros((height, width))

                    for result in task["annotations"][0]["result"]:
                        mask = brush.decode_rle(result["value"]["rle"]).reshape((height, width, 4))[:, :, 3]
                        
                        
                        label = result["value"]["brushlabels"][0]
                        mask_by_clas_dict[label] = np.logical_or(mask, mask_by_clas_dict[label])
                       


            combined_mask = np.zeros((height, width), dtype = np.uint8)
            
            for label in sorted_labels:
                combined_mask[mask_by_clas_dict[label].astype(bool)] = self.classes[label]["label"]

            file_path = Path(task["data"]["image"].split("=")[-1])
            filename =  file_path.parts[-1]

            Image.fromarray(combined_mask).save(os.path.join(os.path.abspath(destination_folder), "masks",  filename))
            shutil.copy(self.label_studio_root / file_path, os.path.join(destination_folder, "images", filename))

    #--------------------------------------
    #         LWS Workflow
    #--------------------------------------      

    def get_result_from_LWSContext(self, ctx: LWSContext):

        results = []

        masks = ctx.get_masks()
        #create Labelstudio Result
        class_polygon_dict = {}
        for label, _ in self.classes.items():
            class_polygon_dict[label] = []

        for mask in masks:
            if not  mask["class"] == "object":
                class_polygon_dict[mask["class"]].append(mask["mask"])

        for label, _ in self.classes.items():
            masks = class_polygon_dict[label]
            if len(masks) > 0:
                combined = masks[0]
            
                for m in masks[1:]:
                    combined = np.logical_or(combined, m)
                combined = (combined > 0).astype(np.uint8) * 255
                results.append(self.masktolabelstudioresult(combined, label))
        
        return results

    def label_with_sam(self, run):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if not self.project_exits(run, "brush"):
           self.add_project(run, "brush")
        else:
            brush_project = ls.get_project(self.runs[run]["brush_project"])
        
        tasks = brush_project.get_tasks()
        ctx = LWSContext()
        
        for task in tasks:
            
            ctx.clear_masks()
            if ctx.ended ==True:
                return

            if len(task["predictions"]) > 0:
                pass
            else:

                results = []
                image = Image.open(Path("data") / Path(task["data"]["image"].split("=")[-1]))
                ctx.set_image(image)
                label = LabelwithSam(ctx)
                ctx = label.get_context()
                del label

                results = self.get_result_from_LWSContext(ctx)
                
                brush_project.create_prediction(
                task_id= task["id"],
                model_version=None,
                result= results
                )

    #doesnt work right           
    def revise_with_sam(self,run, ids):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        if not self.project_exits(run, "brush"):    
            raise RuntimeError("Brush Project doesnt exist")      
        else:
            brush_project = ls.get_project(self.runs[run]["brush_project"])

        ctx = LWSContext()
        for id in ids:
            task = brush_project.get_task(id)
            
            img = Image.open(self.lpath_to_path(task["data"]["image"]))         
            ctx.set_image(img)

            predictions = task["predictions"]
            width, height = img.size

            for prediction in predictions:
                for result in prediction["result"]:
                    mask = {"mask": brush.decode_rle(result["value"]["rle"]).reshape((height, width, 4))[:, :, 3], 
                            "class": self.classes[result["value"]["brushlabels"][0]]["label"]}
                    mask["mask"] = (mask["mask"] != 0).astype(np.uint8)
                    ctx.set_mask( mask)
            
            LWS = LabelwithSam(ctx)

            ctx = LWS.get_context()
            
            if ctx.ended:
                return

            results = self.get_result_from_LWSContext(ctx)
            brush_project.create_prediction(
            task_id= id,
            model_version=None,
            result= results
            )

        

def save_json(root):

    root = Path(root)
    runs = {"runs": {1: {"polygon_project": None, "brush_project": None, "images": [str(file) for file in (root / "run1").iterdir() if str(file).endswith(".png")] },
                2: {"polygon_project": None, "brush_project": None,  "images": [str(file) for file in (root / "run2").iterdir()if str(file).endswith(".png")]},
                3: {"polygon_project": 7, "brush_project": 13,  "images": [str(file) for file in (root / "run3").iterdir()if str(file).endswith(".png")]},
                4: {"polygon_project": None, "brush_project": 23,  "images":[str(file) for file in (root / "run4").iterdir()if str(file).endswith(".png")]},
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


if __name__ == "__main__":
    
    project = label_project()
    project.export_final_masks(6, "dataset/runs/run6" )


    
