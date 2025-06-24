import random
import os 
from pathlib import Path
import shutil
import itertools

import torch

import json
import yaml

from train import train
import lib.Deeplab.network as network

class Options():

    def __init__(self):

        clases = {"backgroud": 0, "cow": 1, "heu": 2}

        #model
        self.model = "deeplabv3plus_resnet50"
        self.num_classes = 3
        self.output_stride = 8

        self.dataset = "Cow_segmentation"
        self.save_val_results = False # True

        #constants
        self.val_interval = 30
        self.total_itrs = 2000
        self.val_batch_size = 10
        self.loss_type = 'cross_entropy'
        self.max_decrease = 0.3

        #hyperparameter
        self.batch_size = 15
        self.class_weights = torch.tensor([0.2, 1.0, 1.0], device="cuda")
        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001

        #visualize Option
        self.enable_vis = None

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.ckpt= None 
        self.continue_training = False

class Crossvalidation:

    def __init__(self, dataset_root):
        
        self.root = Path(dataset_root)
        self.param_grid = {
            "batch_size" : [3, 7, 13],
            "output_stride" : [8,16], 
            "background_weighting" :[0.05, 0.15, 0.3, 0.5], 
            "lerning_rate" : [0.001, 0.01, 0.005, 0.05],
            "weight_decay" : [0.05, 0.02, 0.01, 0.005,0.002,  0.001],
            "loss": ["focal_loss", "cross_entropy"]
        }

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
            
    def cross_validation(self, folds_path):
    
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        grid_org = [dict(zip(keys, v)) for v in itertools.product(*values)]
        total_com = len(grid_org)
        
        to_remove_total = 30

        folds_path = Path(folds_path)
        for path in folds_path.iterdir():
            
            if not (path/ "grid.json" ).is_file():
                
                with open(path/  "grid.json", "w") as file:
                    json.dump(grid_org, file , indent= 4)
            
            with open(path/ "grid.json", "r") as file:
                grid = json.load(file)
            
            already_removed = total_com - len(grid)
            num_picks = max(0, to_remove_total - already_removed)
            self.hyperparameter_optimization(grid, num_picks, path)

    @staticmethod
    def run_config(opts, destination_folder):

        path = Path(destination_folder)
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

def test_config():
    cv = Crossvalidation("dataset")
    opts = Options()
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
