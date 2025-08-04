import random
import os 
from pathlib import Path
import shutil
<<<<<<< HEAD
=======
import itertools
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

import torch

import json
import yaml

from train import train
<<<<<<< HEAD

import optuna
from functools import partial
=======
import lib.Deeplab.network as network
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

class Options():

    def __init__(self):

        clases = {"backgroud": 0, "cow": 1, "heu": 2}

        #model
        self.model = "deeplabv3plus_resnet50"
        self.num_classes = 3
<<<<<<< HEAD
        
=======
        self.output_stride = 8

>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
        self.dataset = "Cow_segmentation"
        self.save_val_results = False # True

        #constants
<<<<<<< HEAD
        self.val_interval = 50
        self.total_itrs = 2000
        self.val_batch_size = 10
        self.max_decrease = 0.55
=======
        self.val_interval = 30
        self.total_itrs = 2000
        self.val_batch_size = 10
        self.loss_type = 'cross_entropy'
        self.max_decrease = 0.3
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

        #hyperparameter
        self.batch_size = 15
        self.class_weights = torch.tensor([0.2, 1.0, 1.0], device="cuda")
        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001
<<<<<<< HEAD
        self.loss_type = 'cross_entropy'
        self.freeze_backbone = False
        self.output_stride = 8
=======

        #visualize Option
        self.enable_vis = None
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
<<<<<<< HEAD
        self.save_param = False
        self.ckpt= None 
        self.continue_training = False


def make_path(opts):
    params = [
        f"bt={opts.batch_size}", 
        f"str={opts.output_stride}",
        f"cw={round(float(opts.class_weights[0]), 4)}", 
        f"wd={round(opts.weight_decay, 6)}", 
        f"lr={round(opts.lr, 6)}",
        f"l={opts.loss_type}", 
        f"lp={opts.lr_policy}",
        f"fb={opts.freeze_backbone}"
    ]

    return "_".join(params)
=======
        self.ckpt= None 
        self.continue_training = False

>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
class Crossvalidation:

    def __init__(self, dataset_root):
        
        self.root = Path(dataset_root)
<<<<<<< HEAD

    @staticmethod
    def clear_directory(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
=======
        self.param_grid = {
            "batch_size" : [3, 7, 13],
            "output_stride" : [8,16], 
            "background_weighting" :[0.05, 0.15, 0.3, 0.5], 
            "lerning_rate" : [0.001, 0.01, 0.005, 0.05],
            "weight_decay" : [0.05, 0.02, 0.01, 0.005,0.002,  0.001],
            "loss": ["focal_loss", "cross_entropy"]
        }
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

    #creates k folds with val train split
    def create_folds(self, k_folds, option = "folds_in_run"):

        run_path = self.root /"runs"

        folds_path = Path(f"dataset/folds")
        #data needs to be temporaly ordered
        if option == "folds_in_run":
            
            folds_path.mkdir(exist_ok=True)
            self.clear_directory(folds_path)
            
            for i in range(k_folds):
                train_paths = []
                eval_paths = []

                fold_path = folds_path / f"{i}"
                fold_path.mkdir(parents=True, exist_ok=True)
                for run in run_path.iterdir():
                    
                    frame_path = (run/ "images")
                    frames = os.listdir(frame_path)
                    n = len(frames)
                    chunk_size = int(n / k_folds)
                    # Get validation indices for this fold
                    val_start = i * chunk_size
                    val_end = val_start + chunk_size if i < k_folds - 1 else n 
                    val_frames = frames[val_start:val_end]

                    #to ensure no temporal overlap
                    if i == 0:
                        val_frames = val_frames[:len(val_frames) - int(0.2 * len(val_frames))]
                    elif i == k_folds -1:
                        val_frames = val_frames[int(0.2 * len(val_frames)):]
                    else:
                        val_frames = val_frames[int(0.1 * len(val_frames)):len(val_frames) -int(0.1 * len(val_frames))]

                    train_frames =frames[:val_start] +frames[val_end:]
                    train_paths.extend([run /"images"/ train_frame for train_frame in train_frames])
                    eval_paths.extend([ run /"images"/ val_frame for val_frame in val_frames])
                
                with open(fold_path / "train.txt", "w") as f:
                    for path in train_paths:
                        f.write(f"{path}\n")

                with open(fold_path / "eval.txt" , "w") as f:
                    for path in eval_paths:
                        f.write(f"{path}\n")

        #safest version for no temporal overlap
        elif option == 'run_as_fold':

            runs = os.listdir(run_path)
            folds_path.mkdir(exist_ok=True)
            self.clear_directory(folds_path)

            for i in range(len(runs)):
                train_runs = []
                eval_runs = []

                train_paths = []
                eval_path = []

                fold_path = folds_path / f"{i}"

                for j in range(len(runs)):

                    if j==i:
                        eval_runs.append(runs[j])
                    else:
                        train_runs.append(runs[j])

                for run in train_runs:
                    print("todo")
            
        else:
            raise RuntimeError("not a valid Option")
            
    def cross_validation(self):
<<<<<<< HEAD

        folds_path = self.root / "folds"
        
        for path in folds_path.iterdir():
            
            self.hyperparameter_optimization(path)

    
    def objective(self, trial, destination_folder):
        opts = Options()

        opts.output_stride = trial.suggest_int('output_stride', 8, 16, step = 8)
        opts.batch_size = trial.suggest_int("batch_size", 3, 15, step = 1)
        opts.lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log= True)
        opts.class_weights[0] = trial.suggest_float("bg_weight", 0.1, 1)
        opts.loss_type = trial.suggest_categorical("loss",  ["focal_loss", "cross_entropy"])
        opts.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        opts.lr_policy=  trial.suggest_categorical("lr_policy",  ["poly", "step"])
        opts.freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])

        metrics = self.run_config(opts, destination_folder)
        metrics =  max(metrics["validation"], key=lambda d: d["Class IoU"]["0"] + d["Class IoU"]["1"]+ d["Class IoU"]["2"])
        return 1 -(1/3 * metrics["Class IoU"]["0"] + 1/3 * metrics["Class IoU"]["1"]+ 1/3 * metrics["Class IoU"]["2"])

    def hyperparameter_optimization(self, path):

        study = optuna.create_study(direction='minimize')
        study.optimize(partial(self.objective, destination_folder=path), n_trials=40)

        with open(path/ "best.json", "w") as f:
            json.dump(study.best_params, f)
=======
    
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        grid_org = [dict(zip(keys, v)) for v in itertools.product(*values)]
        total_com = len(grid_org)
        
        to_remove_total = 30

        folds_path = self.root / "folds"
        ""
        for path in folds_path.iterdir():
            
            if not (path/ "grid.json" ).is_file():
                
                with open(path/  "grid.json", "w") as file:
                    json.dump(grid_org, file , indent= 4)
            
            with open(path/ "grid.json", "r") as file:
                grid = json.load(file)
            
            already_removed = total_com - len(grid)
            num_picks = max(0, to_remove_total - already_removed)
            self.hyperparameter_optimization(grid, num_picks, path)
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

    @staticmethod
    def run_config(opts, destination_folder):

        path = Path(destination_folder)
<<<<<<< HEAD
        config_path = path / make_path(opts)

        if (path/"index.json").is_file():

            with open(path/"index.json", "r") as f:
                index = json.load(f)
            
            index.append(str(config_path))

            with open(path/"index.json", "w") as f:
                json.dump(index, f, indent=4)

        else:
            index = [str(config_path)]
            with open(path/"index.json", "w") as f:
                json.dump(index, f, indent=4)

        (config_path).mkdir(parents=True, exist_ok=True)
        with open(config_path /"config.yaml", "w") as f:
            yaml.safe_dump({
                "batchsize": opts.batch_size,
                "output_stride": opts.output_stride,
                "background_weighting": round(float(opts.class_weights[0]), 4),
                "learning_rate": round(opts.step_size, 4), 
                "weight_decay": round(opts.weight_decay, 4), 
                "loss_type": opts.loss_type,
                "lr_policy": opts.lr_policy,
                "freeze_backbone": opts.freeze_backbone
            }, f)
        train(opts, config_path)
        
        with open(config_path / "metrics.json", "r") as f:
            metrics = json.load(f)

        return metrics

=======
        config_path = path/ f"bt={opts.batch_size}_str={opts.output_stride}_cw={round(float(opts.class_weights[0]), 4)}_lr={round(opts.step_size, 4)}_wd={round(opts.weight_decay, 4)}_l={opts.loss_type}"

        if config_path.is_dir() and (path / f'latest.pth').is_file():

                checkpoint_path = path / f'latest.pth'
                checkpoint = torch.load(checkpoint_path , map_location=torch.device('cpu'), weights_only=False)

                if checkpoint["cur_itrs"] <= opts.total_itrs:
                    opts.ckpt = checkpoint_path
                    opts.continue_training = True
                    train(opts, config_path)

        else:

            if (path/"index.json").is_file():

                with open(path/"index.json", "r") as f:
                    index = json.load(f)
                
                index.append(str(config_path))

                with open(path/"index.json", "w") as f:
                    json.dump(index, f, indent=4)

            else:

                index = [str(config_path)]
                with open(path/"index.json", "w") as f:
                    json.dump(index, f, indent=4)

            (config_path).mkdir(parents=True, exist_ok=True)
            with open(config_path /"config.yaml", "w") as f:
                yaml.safe_dump({
                    "batchsize": opts.batch_size,
                    "output_stride": opts.output_stride,
                    "background_weighting": round(float(opts.class_weights[0]), 4),
                    "learning_rate": round(opts.step_size, 4), 
                    "weight_decay": round(opts.weight_decay, 4), 
                    "loss_type": opts.loss_type
                }, f)
            train(opts, config_path)

    #random search
    def hyperparameter_optimization(self, grid, num_picks, fold_path):

        for i in range(num_picks):
            opts = Options()

            index = random.randrange(len(grid))
            hyperparameters = grid[index]
            opts.batch_size =  hyperparameters["batch_size"]
            opts.output_stride =hyperparameters["output_stride"]
            opts.class_weights[0] = hyperparameters[ "background_weighting"]
            opts.step_size = hyperparameters["lerning_rate"]
            opts.weight_decay = hyperparameters["weight_decay"]
            opts.loss_type = hyperparameters["loss"]

            self.run_config(opts, fold_path)

            grid.pop(index)
            with open(fold_path/  "grid.json", "w") as file:
                json.dump(grid, file , indent= 4)
    
    @staticmethod
    def clear_directory(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f

def test_config():
    cv = Crossvalidation("dataset")
    opts = Options()
<<<<<<< HEAD
    opts.save_param = True
    opts.batch_size = 4
    opts.output_stride = 8
    opts.lr = 0.005608581136546391
    opts.weight_decay =  0.00044588318263961654
    opts.lr_policy = "poly"
    opts.class_weights[0] =  0.14441240416574808
    opts.loss_type = "focal_loss"
    opts.freeze_backbone = True

    opts.total_itrs = 10000
    opts.val_interval = 10
    opts.val_batch_size = 10
    opts.save_val_results = False
    

    cv.run_config(opts, "dataset/test")

if __name__ == "__main__":

    #cv = Crossvalidation("dataset")
    test_config()
=======
    opts.batch_size = 13
    opts.output_stride = 16
    opts.class_weights[0] = 0.4
    opts.class_weights[2] = 1.2
    opts.total_itrs = 10000
    opts.val_interval = 20
    opts.val_batch_size = 10
    opts.loss_type = "cross_entropy"

    cv.run_test(opts, "dataset/test")

if __name__ == "__main__":

    cv = Crossvalidation("dataset")
    cv.cross_validation()
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
