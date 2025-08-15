from pathlib import Path  
import json
import yaml
import os

import lib.Deeplab.network as network
from lib.Deeplab.metrics import StreamSegMetrics
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils import data

from cross_validation import Options
from train import train, validate
from my_dataset import Mydataset

class Evaluation:

    def __init__(self, crossvalidation_root):

        self.root = Path(crossvalidation_root)
        self.studies = []

        self.metrics = ["Overall Acc","Mean Acc","FreqW Acc","Mean IoU"]

        for fold in self.root.iterdir():
            study = optuna.load_study(study_name=f"fold_optimization",
                                storage=f"sqlite:///{fold}/optuna_study.db")
            self.studies.append(study.trials_dataframe())

        


    def plot_optimization(self, fold: int):

        trials = self.studies[fold]
        completed_trials = trials[trials['state'] == 'COMPLETE']
        values = completed_trials['value']
        x = np.arange(len(values))
        
        plt.plot(completed_trials['number'], values)
        plt.show()

    def plot_best_config(self, fold):
        
        with open(self.root / f"{fold}"/ "index.json", 'r') as f:
            index = json.load(f)

        trials = self.studies[fold]
        sorted_trials = trials.sort_values(by='value')
        best = sorted_trials.iloc[0]

        path = Path(index[best['number']])
        
        with open(path /"metrics.json", 'r') as f:
            metrics = json.load(f)

        self.plot_training(metrics)
    
    def plot_training(self, metrics):
        
        train_loss = [m["loss"] for m in metrics['train']]
        train_x = [m["itr"] for m in metrics['train']]

        for metric in self.metrics:
            eval_metric = [1 -m[metric] for m in metrics['validation']]
            eval_x =  [m['itr'] for m in metrics['validation']]
            plt.plot(eval_x, eval_metric, label= metric)

        plt.plot(train_x, train_loss)
        plt.legend()
        plt.show()
    
    def train_best_config(self, fold):
        
        opts = Options()

        with open(self.root / f"{fold}"/ "index.json", 'r') as f:
            index = json.load(f)

        trials = self.studies[fold]
        sorted_trials = trials.sort_values(by='value')
        best = sorted_trials.iloc[0]
        path = Path(index[best['number']])

        with open(path / "config.yaml", "r") as f:
            params = yaml.safe_load(f)
        
        opts.from_dict(params)
        opts.save_param = True
        opts.continue_training = False
        opts.total_epochs = 100
        opts.save_val_results = True
        config_path = self.root / f"{fold}"/ "best"
        config_path.mkdir(exist_ok=True)
        train(opts, config_path)
    
    def test_best_models(self, fold):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        opts = Options()
        opts.save_val_results = True
        opts.ckpt =Path(f"dataset/folds/{fold}/best/best.pth")
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)

        model.eval()
        metrics = StreamSegMetrics(opts.num_classes)

        test_frames = [Path("dataset/test/images") / name for name in   os.listdir("dataset/test/images")]
        test_dst  = Mydataset(test_frames,preload = True)
        test_loader = data.DataLoader(test_dst, batch_size=1, shuffle=False,
        drop_last=True, pin_memory=True) 

        val_score, _ = validate(opts, model, test_loader, device, metrics, Path(f"dataset/folds/{fold}/best"))
        print(val_score)


if __name__ == "__main__":

    eval = Evaluation("dataset/folds")
    eval.train_best_config(2)
    eval.test_best_models(2)


    






        



