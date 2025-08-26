import random
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import shutil
import json
import yaml
from functools import partial

from train import train
from util.options import Options

import optuna

class Training:

    """
    End-to-end workflow for cross-validation and hyperparameter optimization.

    Steps
    -----
    1. create_folds(k, option): make train/val splits, write train.txt/eval.txt.
    2. cross_validation(): loop over folds, run hyperparameter search.
    3. hyperparameter_optimization(path): Optuna study per fold.
    4. objective(trial, path): define search space, run one config.
    5. run_config(opts, path): save config, call train(), load metrics.
    """

    def __init__(self, dataset_root):
        """
        Initialize the workflow object with a dataset root.

        Parameters
        ----------
        dataset_root :
            Path to the root directory of the dataset. This directory is expected
            to contain a `runs/` subdirectory

        Preconditions
        -------------
        - Downstream methods (`create_folds`) assume the
        following structure under `dataset_root`:
            dataset_root/
                runs/
                    <run1>/images/*.png 
                    ...
        """

        self.root = Path(dataset_root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        # check runs/
        runs_path = self.root / "runs"
        if not runs_path.exists():
            raise FileNotFoundError(f"Missing required subdirectory: {runs_path}")
        if not runs_path.is_dir():
            raise NotADirectoryError(f"`runs` is not a directory: {runs_path}")

        runs = list(runs_path.iterdir())
        if not runs:
            raise FileNotFoundError(f"No runs found under {runs_path}")

        bad_runs = []
        for run in runs:
            if not (run / "images").exists():
                bad_runs.append(run)

    @staticmethod
    def clear_directory(path):
        """
        Remove all files and subdirectories from a directory.

        Parameters
        ----------
        path : str or Path
            Path to the directory to clear.
        """
        
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)


    def create_folds(self, k_folds: int, option: str = "folds_in_run"):
        """
        Create cross-validation folds for temporally ordered image runs.

        Two splitting strategies are supported:

        1. option = "folds_in_run"
        Each run is internally split into `k_folds` contiguous chunks of frames.
        For each fold:
            - One chunk is held out for validation (chosen consistently per run with a seed).
            - The remaining chunks are used for training.
        Small overlaps near chunk boundaries are trimmed to reduce
        temporal leakage between train/val.

        Output: dataset/folds/{i}/train.txt and eval.txt listing absolute paths
        to training and validation images for fold i.

        2. option = "run_as_fold"
        Each entire run is used as the validation set for one fold, while all
        other runs are used for training (leave-one-run-out CV).
        Currently only stubbed in; training paths need to be written out.

        Parameters
        ----------
        k_folds : int
            Number of folds to generate.
        option : {"folds_in_run", "run_as_fold"}, default="folds_in_run"
            Strategy to generate folds.
        """

        def contiguous_chunk_bounds(n, k):
            base, rem = divmod(n, k)
            bounds = []
            start = 0
            for j in range(k):
                size = base + (1 if j < rem else 0)
                end = start + size
                bounds.append((start, end))
                start = end
            return bounds  

        def trim_for_no_overlap(frames_slice, chunk_pos, k):

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

        run_path = self.root /"runs"

        folds_path = Path(f"dataset/folds")

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
                        r = random.Random(f"{SEED}-{run.name}")
                        perm = list(range(k_folds))
                        r.shuffle(perm)
                        run_permutation[run] = perm

                    chosen_chunk = perm[i]
                    s, e = bounds[chosen_chunk]

                    val_frames = frames[s:e]
                    val_frames = trim_for_no_overlap(val_frames, chosen_chunk, k_folds)

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
            raise RuntimeError("not implemented yet")
            
        else:
            raise RuntimeError("not a valid Option")
            
    def cross_validation(self):
        """
        Run cross-validation by iterating over prepared folds.

        This method expects that dataset folds have already been created
        (via `create_folds`) 
        """

        folds_path = self.root / "folds"
        
        for path in folds_path.iterdir():
            
            self.hyperparameter_optimization(path)

    def hyperparameter_optimization(self, path: Path, num_iteration: int=200):
        """
        Run hyperparameter optimization for a given fold using Optuna.

        This method sets up an Optuna study with a TPE sampler and a median
        pruner, stores the study in a SQLite database inside the given fold
        directory, and optimizes the objective function.

        Parameters
        ----------
        path : pathlib.Path
            Path to the fold directory. Must contain `train.txt` and `eval.txt`
            files generated by `create_folds`.

        Preconditions
        -------------
        -This method expects that dataset folds have already been created
        (via `create_folds`) 

        """

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=1,  
            multivariate=True,
            group=True,
            seed=42,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=25,    
            n_warmup_steps=30      
        )

        study = optuna.create_study(study_name=f"fold_optimization",
                                    storage=f"sqlite:///{path}/optuna_study.db",
                                    load_if_exists=True,
                                    direction='minimize',
                                    sampler=sampler,
                                    pruner=pruner)
        
        study.optimize(partial(self.objective, destination_folder=path), n_trials=num_iteration)

    def objective(self, trial, destination_folder: Path):
        
        """
        Optuna objective function for semantic segmentation hyperparameter search.

        This function defines the hyperparameter search space and starts the Traning for an config

        Parameters
        ----------
        trial : optuna.trial.Trial
            The Optuna trial object used to sample hyperparameters.
        destination_folder : pathlib.Path
            Path to the fold directory where tranings and validation metrics get saved too

        Returns
        -------
        float
            Objective value for Optuna to minimize (`1 - best_mIoU`).
        """

        opts = Options()

        opts.output_stride = trial.suggest_int('output_stride', 16, 16, step = 8)
        opts.batch_size = trial.suggest_int("batch_size", 3, 15, step = 1)
        opts.lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log= True)
        opts.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        opts.loss_type = trial.suggest_categorical("loss",  ["focal_loss", "cross_entropy"])
        if opts.loss_type == "cross_entropy": 
            opts.class_weights[0] = trial.suggest_float("bg_weight", 0.1, 1)

        opts.freeze_backbone =trial.suggest_categorical('freeze_backbone', [True, False])

        opts.lr_policy =  trial.suggest_categorical("lr_policy",  ["step", "poly", "none"])
        
        if opts.lr_policy == "step":
            opts.step_size = trial.suggest_int("step_size", 2, 30)
            opts.gamma = trial.suggest_float("gamma", 0.1, 0.95)
        elif opts.lr_policy == "poly":
            opts.poly_power = trial.suggest_float("power", 0.5, 3.0)
        
        
        metrics = self.run_config(opts, destination_folder,trial= trial)
        metrics =  max(metrics["validation"], key=lambda d: d["Class IoU"]["1"])

        return 1 - metrics["Class IoU"]["1"]

    @staticmethod
    def run_config(opts, destination_folder, trial =None):
        """
        Run a training configuration and return its evaluation metrics.

        This method is responsible for:
        1. Creating a unique directory for the given config.
        2. Recording the config in an `index.json`
        3. Saving the config as `config.yaml`.
        4. Calling the `train` function to perform training and evaluation.

        Parameters
        ----------
        opts : Options
            The configuration object containing hyperparameters and settings.
        destination_folder : str or pathlib.Path
            The base directory where this trial's results should be stored.
            Each config will be placed in a subfolder 
        trial : optuna.Trial, optional
            The Optuna trial object, if th
            is run is part of hyperparameter search.
            Passed through to `train` for pruning integration.

        Returns
        -------
        dict
            Dictionary of metrics loaded from `metrics.json`. Expected to contain
            a `"validation"` key with a list of per-epoch metrics.

        Preconditions
        -------------
        - destination_folder need to have eval.txt and train.txt files in the parant folder 
        """

        path = Path(destination_folder)
        config_path = path / opts.create_dir_name()

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



    

