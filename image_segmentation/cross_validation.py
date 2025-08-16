import random
import os 
from pathlib import Path
import shutil

import torch
import re
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
        self.val_interval =2
        self.total_epochs = 6
        self.val_batch_size = 10
        self.max_decrease = 0.35

        #hyperparameter
        self.batch_size = 15
        self.class_weights = torch.tensor([0.2, 1.0, 1.0], device="cuda")
        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.loss_type = 'cross_entropy'
        self.freeze_backbone = False
        self.output_stride = 8

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.save_param = False
        self.ckpt= None 
        self.continue_training = False

    def to_dict(self):

        return{"batchsize": self.batch_size,
                "output_stride":self.output_stride,
                "background_weighting": round(float(self.class_weights[0]), 6),
                "learning_rate": round(self.lr, 6), 
                "weight_decay": round(self.weight_decay, 6), 
                "loss_type": self.loss_type,
                "lr_policy": self.lr_policy,
                "freeze_backbone": self.freeze_backbone}
    
    def from_dict(self, params):

        self.batch_size = params["batchsize"]
        self.output_stride = params["output_stride"]
        self.class_weights[0] = params["background_weighting"]
        self.lr = params["learning_rate"]
        self.weight_decay = params["weight_decay"]
        self.loss_type = params["loss_type"] 
        self.lr_policy =params["lr_policy"]
        self.freeze_backbone = params["freeze_backbone"]


def contiguous_chunk_bounds(n, k):
    """Return k contiguous chunks that cover range(n) as evenly as possible."""
    base, rem = divmod(n, k)
    bounds = []
    start = 0
    for j in range(k):
        size = base + (1 if j < rem else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds  # list of (start, end), len == k

def trim_for_no_overlap(frames_slice, chunk_pos, k):
    """Trim edges of the validation slice to avoid temporal overlap with train."""
    m = len(frames_slice)
    if m == 0:
        return frames_slice
    if chunk_pos == 0:
        end = m - int(0.2 * m)
        return frames_slice[:max(end, 0)]
    elif chunk_pos == k - 1:
        start = int(0.2 * m)
        return frames_slice[start:]
    else:
        start = int(0.1 * m)
        end = m - int(0.1 * m)
        if end <= start:
            return []
        return frames_slice[start:end]

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
class Training:

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
            
            SEED = 12345
            run_permutation = {} 
            for i in range(k_folds):
                train_paths = []
                eval_paths = []

                fold_path = folds_path / f"{i}"
                fold_path.mkdir(parents=True, exist_ok=True)

                for run in run_path.iterdir():
                    frame_path = run / "images"
                    if not frame_path.exists():
                        continue

                    frames = sorted(os.listdir(frame_path))
                    n = len(frames)
                    if n == 0:
                        continue

                    bounds = contiguous_chunk_bounds(n, k_folds)
                    perm = run_permutation.get(run)
                    if perm is None:
                        # fallback in case run was added later
                        r = random.Random(f"{SEED}-{run.name}")
                        perm = list(range(k_folds))
                        r.shuffle(perm)
                        run_permutation[run] = perm

                    # choose which chunk is eval for THIS fold for THIS run
                    chosen_chunk = perm[i]
                    s, e = bounds[chosen_chunk]

                    val_frames = frames[s:e]
                    val_frames = trim_for_no_overlap(val_frames, chosen_chunk, k_folds)

                    # everything outside chosen [s:e] is train
                    train_frames = frames[:s] + frames[e:]

                    train_paths.extend([frame_path / f for f in train_frames])
                    eval_paths .extend([frame_path / f for f in val_frames])

                with open(fold_path / "train.txt", "w", encoding="utf-8") as f:
                    for path in train_paths:
                        f.write(f"{path}\n")

                with open(fold_path / "eval.txt", "w", encoding="utf-8") as f:
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
        opts.lr_policy =  trial.suggest_categorical("lr_policy",  ["none"])
        opts.freeze_backbone =trial.suggest_categorical('freeze_backbone', [True, False])

        metrics = self.run_config(opts, destination_folder,trial= trial)
        metrics =  max(metrics["validation"], key=lambda d: d["Mean IoU"])
        return 1 - metrics["Mean IoU"]

    def hyperparameter_optimization(self, path):
        

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=25,  
            multivariate=True,
            group=True,
            seed=42,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=25,    
            n_warmup_steps=20      
        )

        study = optuna.create_study(study_name=f"fold_optimization",
                                    storage=f"sqlite:///{path}/optuna_study.db",
                                    load_if_exists=True,
                                    direction='minimize',
                                    sampler=sampler,
                                    pruner=pruner)
        
        study.optimize(partial(self.objective, destination_folder=path), n_trials=200)

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
            yaml.safe_dump(opts.to_dict(), f)
        train(opts, config_path, trial=trial)
        
        with open(config_path / "metrics.json", "r") as f:
            metrics = json.load(f)

        return metrics


def test_config():

    cv = Training("dataset")
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

    cv = Training("dataset")
    cv.cross_validation()



    

