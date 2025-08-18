import json
from pathlib import Path
import os
from tqdm import tqdm
import shutil
import datetime


from label_studio_sdk import Client
import label_studio_converter.brush as brush

import numpy as np

from PIL import Image
from LabelwithSam import LabelwithSam, LWSContext

#starte label studio mit LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
#create a dataset mit dem Namen dataset1 sub dir von <root> und füge als storage hinzu

#sync up mit Labelstudio images
#wichitg filepath darf nicht mit der labaelstudio root anfangen sonder erst dannach
#self.root muss subdir von LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT= <ROOT> sein

#class for labeling and managing working with Labelstudio
class label_project:

    def __init__(self, root_path = "data/data_set"):
        
        """
        Manage a Label Studio labeling project and local dataset sync.

        This class handles:
        - Connecting to a Label Studio server (API key + host).
        - Managing project progress and configurations (stored in progress.json).
        - Ensuring Label Studio is correctly set up for local file serving.
        - Tracking runs, backup data, and semantic class definitions.

        Requirements
        ------------
        - Environment variables must be set:
            * LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<ROOT>
            * LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
        - The project root must be a subdirectory of <ROOT>.
        - If no `progress.json` is found, API key and host are requested interactively.
        """

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
                        "brush_config": """<View>
                        <Image name="image" value="$image" zoom="true"/>
                        <BrushLabels name="tag" toName="image">
                        <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff0080"/><Label value="object" background="#FFC069"/><Label value="cow_head" background="#400080"/></BrushLabels>
                        </View>""",
                        }
            
        self.runs = progress["runs"]
        self.key = progress["api_key"]
        self.host = progress["host"]
        self.config = progress["brush_config"]

        self.save_config()
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

        self.classes = {"heu": {"label":2, "prio": 0}, "cow": {"label":1, "prio": 1}, "cow_head": {"label": 3, "prio": 2}}

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

    def add_run(self, image_folder: str, run):
        """
        Add a new labeling run by importing a folder of PNG images.

        This method assigns a unique incremental key to the new run,
        creates a dedicated directory under the project root to store
        the run's images, copies all PNG files from the specified folder,
        and updates the internal run configuration.

        Args:
            image_folder (str):
                Path to the directory containing PNG images to be added as a new run.

        Side Effects:
            - Creates a new directory under self.root named 'runN', where N is the next available integer key.
            - Copies all '.png' files from the input folder into the new run directory.
            - Updates self.runs to include the new run entry with the list of copied image paths.
            - Persists the updated configuration by calling self.save_config()."""

        if str(run) in self.runs.keys():
            raise RuntimeError("This run already exists")

        image_folder = Path(image_folder)

        run_path =(self.root / f"run{run}")
        run_path.mkdir(parents=True, exist_ok=True)
        
        ls = Client(url=self.host, api_key=self.key)
        project = ls.start_project(
                title = f"run_{str(run)}",
                label_config = self.config
            )

        self.runs[str(run)] = {"project_id": project.id} 
        self.save_config()

        for file in (image_folder).iterdir():
            if file.suffix == ".png":
                shutil.copy(file, str(run_path/ file.name))

                path = run_path / file.name
                project.import_tasks({
                "data": {
                    "image": f"/data/local-files/?d={str(Path(*path.parts[1:]))}"
                    }})
        
        """cwd = Path(os.getcwd())
        storage_config = {
            'path': f'{str(cwd / self.root)}',  
            'use_blob_urls': True,                    
            'title': 'My Local Storage',
            'description': 'Local dataset for image labeling',
            'regex_filter': r'.*\.(jpg|png|jpeg)$',   
            'use_file_name_as_label': False
        }

        # Create local file import storage
        response = ls.make_request(
            'post',
            '/api/storages/localfiles',
            json=storage_config
        )"""

        self.save_config()
    

    def fix(self, run):
        run = str(run)
        ls = Client(url=self.host, api_key=self.key)
        project = ls.get_project(self.runs[run][f"project_id"])

        self.runs[str(run)] = {"project_id": project.id} 
        self.save_config()

        task = project.get_tasks()

        project.delete_all_tasks()

        for tasks in task:
            path = tasks["data"]["image"]
            path = self.lpath_to_path(path)
            path =Path("data")/ path.relative_to(self.label_studio_root)
            tasks["data"]["image"] = f"/data/local-files/?d={str(path)}"
            
            project.import_tasks(tasks)


    #danger zone deletes a run from the progress json and the images get deleted
    def delete_run(self, run: int):

        if os.path.exists(self.root / f"run{run}"):
            shutil.rmtree(self.root / f"run{run}")
        self.runs.pop(str(run), None)
        self.save_config()

    #nicht getestet
    def reset_tasks(self, run):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        project = ls.get_project(self.runs[run][f"project_id"])
        self.backup_run(run)
        project.delete_all_tasks()

        for image in (self.root / f"run{run}").iterdir():
            path = Path(*Path(image).parts[1:])
            project.import_tasks({
            "data": {
                "image": f"/data/local-files/?d={str(path)}"
                }})
            
    #saves the current config of the project to the progress json
    def save_config(self):

        with open(self.root / "progress.json", "w") as f:
            json.dump({"runs": self.runs,
                        "api_key": self.key,
                        "host": self.host,
                        "brush_config": self.config}, f, indent=4)

    #saves the actual progress for all runs in from of the labelstudio jsons to the backup folder
    def backup_progress(self):

        for run, _ in self.runs.items():
            self.backup_run(run)

    #saves the actual progress for one run in form of the labelstudio jsons to the backup folder
    def backup_run(self, run):

        ls = Client(url=self.host, api_key=self.key)

        project = ls.get_project(self.runs[run]["project_id"]) 
        project_tasks = project.get_tasks()

        backup_run_path = self.backup_path / f"run{run}_backup.json"

        with open(backup_run_path, "w") as f:
            json.dump(project_tasks, f)

        print(f"Backup von run {run} erstellt")
    

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

    #meant for label unlabeles data
    def label_with_sam(self, run):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        brush_project = ls.get_project(self.runs[run]["project_id"])
        
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

    #works on predictions and can make annotations too
    def revise_with_sam(self,run, predic, ids = None):

        run = str(run)
        ls = Client(url=self.host, api_key=self.key)

        brush_project = ls.get_project(self.runs[run]["project_id"])

        if ids is None:
            tasks = brush_project.get_tasks()
            ids = [task["id"] for task in tasks]

        ctx = LWSContext()
        for id in ids:
            ctx.clear_masks()
            task = brush_project.get_task(id)
            
            img = Image.open(self.lpath_to_path(task["data"]["image"]))         
            ctx.set_image(img)

            if predic:
                data = task["predictions"]
            else:
                data = task["annotations"]

            if len(data) > 0:
                prediction = data[-1]
                width, height = img.size

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
            if predic:
                brush_project.create_prediction(
                task_id= id,
                model_version=None,
                result= results
                )
            else:
                brush_project.create_annotation(
                task_id= id,
                result= results
                )

    #exports all brush labled masks
    #uses last annotation
    def export_final_masks(self, run: int, destination_folder):

        VGA_SIZE = (640, 480)

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


        project = ls.get_project(self.runs[run]["project_id"]) 

        
        tasks = project.get_tasks()
        labels = list(self.classes.keys())
        sorted_labels = sorted(labels, key=lambda l: self.classes[l]["prio"], reverse = False)

        for task in tqdm(tasks):
            
            if len(task["annotations"]) > 0:
                if len(task["annotations"][-1]["result"]) >0:
                    mask_by_clas_dict = {}
                    height = task["annotations"][0]["result"][0]["original_height"]
                    width = task["annotations"][0]["result"][0]["original_width"]

                    for label, _ in self.classes.items():
                        mask_by_clas_dict[label] = np.zeros((height, width))

                    for result in task["annotations"][-1]["result"]:
                        mask = brush.decode_rle(result["value"]["rle"]).reshape((height, width, 4))[:, :, 3]
                        
                        label = result["value"]["brushlabels"][0]
                        if not label =="cow_head":
                            mask_by_clas_dict[label] = np.logical_or(mask, mask_by_clas_dict[label])
                       

            combined_mask = np.zeros((height, width), dtype = np.uint8)
            
            for label in sorted_labels:
                combined_mask[mask_by_clas_dict[label].astype(bool)] = self.classes[label]["label"]

            file_path = Path(task["data"]["image"].split("=")[-1])
            filename =  file_path.parts[-1]

            
            mask_img = Image.fromarray(combined_mask).resize(VGA_SIZE, Image.NEAREST)  # NEAREST to preserve mask labels
            mask_path = os.path.join(os.path.abspath(destination_folder), "masks", filename)
            mask_img.save(mask_path)

            # Resize image
            image_path = self.label_studio_root / file_path
            image_img = Image.open(image_path).resize(VGA_SIZE, Image.BILINEAR)  # BILINEAR for smoother image resize
            image_save_path = os.path.join(destination_folder, "images", filename)
            image_img.save(image_save_path)

def save_json(root):

    root = Path(root)
    runs = {"runs": {1: {"project_id": 3},
                3: {"project_id": 13},
                4: {"project_id": 23},
                5: {"project_id": 9},
                6: {"project_id": 14},
                7: {"project_id": 24}
                },
            "api_key": "179e598bf40ebfc8904c1987e2507c0fe11936f0",
            "host": "http://localhost:8080",
            "brush_config" : """<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
  <Label value="cow" background="#FFA39E"/><Label value="heu" background="#ff0080"/><Label value="object" background="#FFC069"/><Label value="cow_head" background="#400080"/></BrushLabels>
</View>"""
            }
    
    with open(root /"progress.json", "w") as f:
        json.dump(runs, f, indent=4)


if __name__ == "__main__":
    
    project = label_project()
    project.export_final_masks(1, "dataset/test")


    
