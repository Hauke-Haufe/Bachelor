from pathlib import Path  
import matplotlib.pyplot as plt
import numpy as np
import optuna
import json
import yaml
import re

from cross_validation import Options
from train import train

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
        opts.total_itrs = 10000
        config_path = self.root / f"{fold}"/ "best"
        config_path.mkdir(exist_ok=True)
        train(opts, config_path)
    

if __name__ == "__main__":

    eval = Evaluation("dataset/folds")
    eval.train_best_config(0)


    






        



