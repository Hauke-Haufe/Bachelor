import random
import os 
from pathlib import Path
import shutil

import torch

import json
import yaml

from train import train

import optuna
from functools import partial

class Options():

    def __init__(self):

        clases = {"backgroud": 0, "cow": 1, "heu": 2}

        #model
        self.model = "deeplabv3plus_resnet50"
        self.num_classes = 3
        
        self.dataset = "Cow_segmentation"
        self.save_val_results = False # True

        #constants
        self.val_interval = 50
        self.total_itrs = 300
        self.val_batch_size = 10
        self.max_decrease = 0.35

        #hyperparameter
        self.batch_size = 15
        self.class_weights = torch.tensor([0.2, 1.0, 1.0], device="cuda")
        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001
        self.loss_type = 'cross_entropy'
        self.freeze_backbone = False
        self.output_stride = 8

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
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
class Crossvalidation:

    def __init__(self, dataset_root):
        
        self.root = Path(dataset_root)

    @staticmethod
    def clear_directory(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)

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
                    frames = sorted(os.listdir(frame_path))
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
        opts.freeze_backbone =trial.suggest_categorical('freeze_backbone', [True, False])

        metrics = self.run_config(opts, destination_folder,trial= trial)
        metrics =  max(metrics["validation"], key=lambda d: d["Mean IoU"])
        return 1 - metrics["Mean IoU"]

    def hyperparameter_optimization(self, path):

        study = optuna.create_study(study_name=f"fold_optimization",
                                    storage=f"sqlite:///{path}/optuna_study.db",
                                    load_if_exists=True,
                                    direction='minimize',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=20))
        
        study.optimize(partial(self.objective, destination_folder=path), n_trials=250)

        with open(path/ "best.json", "w") as f:
            json.dump(study.best_params, f)

        df = study.trials_dataframe()
        df.to_csv(path/"optuna_trials.csv", index=False)

    @staticmethod
    def run_config(opts, destination_folder, trial =None):

        path = Path(destination_folder)
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
                "background_weighting": round(float(opts.class_weights[0]), 6),
                "learning_rate": round(opts.step_size, 6), 
                "weight_decay": round(opts.weight_decay, 6), 
                "loss_type": opts.loss_type,
                "lr_policy": opts.lr_policy,
                "freeze_backbone": opts.freeze_backbone
            }, f)
        train(opts, config_path, trial=trial)
        
        with open(config_path / "metrics.json", "r") as f:
            metrics = json.load(f)

        return metrics


def test_config():


    cv = Crossvalidation("dataset")
    opts = Options()
    opts.save_param = True
    opts.batch_size = 14
    opts.output_stride = 16
    opts.lr =  0.005937
    opts.weight_decay =  0.000593
    opts.lr_policy = "step"
    opts.class_weights[0] =  0.4431
    opts.loss_type = "focal_loss"
    opts.freeze_backbone = True

    opts.total_itrs = 10000
    opts.val_interval = 40
    opts.val_batch_size = 10
    opts.save_val_results = True
    
    cv.run_config(opts, "dataset/test")

if __name__ == "__main__":

    cv = Crossvalidation("dataset")
    cv.create_folds(5)
    #cv.cross_validation()

