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
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from cross_validation import Options
from train import train, validate
from my_dataset import Mydataset
import pandas as pd

from torchvision.transforms import v2

class Evaluation:

    def __init__(self, crossvalidation_root):

        self.root = Path(crossvalidation_root)
        self.studies = []

        self.metrics = ["Overall Acc","Mean Acc","FreqW Acc","Mean IoU"]

        for fold in self.root.iterdir():
            if fold.is_dir():
                study = optuna.load_study(study_name=f"fold_optimization",
                                storage=f"sqlite:///{fold}/optuna_study.db")
                self.studies.append(study.trials_dataframe())
        
        self.num_folds = len(self.studies)

    def train_best_config_fold(self, fold):
        opts = Options()

        with open(self.root / f"{fold}"/ "index.json", 'r') as f:
            index = json.load(f)

        trials = self.studies[fold]
        sorted_trials = trials.sort_values(by='value')
        best = sorted_trials.iloc[0]
        path = Path(index[best['number']])

        with open(self.root/ Path(*path.parts[2:])/ "config.yaml", "r") as f:
            params = yaml.safe_load(f)
        
        opts.from_dict(params)
        opts.save_param = True
        opts.continue_training = False
        opts.total_epochs = 100
        opts.save_val_results = True
        config_path = self.root / f"{fold}"/ "best"
        config_path.mkdir(exist_ok=True)
        train(opts, config_path)

    def print_best_configs(self):
        
        for fold in range(self.num_folds):
            opts = Options()

            with open(self.root / f"{fold}"/ "index.json", 'r') as f:
                index = json.load(f)

            trials = self.studies[fold]
            sorted_trials = trials.sort_values(by='value')
            best = sorted_trials.iloc[0]
            path = Path(index[best['number']])

            with open(self.root/ Path(*path.parts[2:])/ "config.yaml", "r") as f:
                params = yaml.safe_load(f)
            
            print(f"Beste Konfiguration in Fold {fold}: {params}")
        
    def train_best_models(self):

        for fold in range(self.num_folds):
            self.train_best_config_fold(fold)
    
    def test_best_models(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        scores = {}
        for fold in range(self.num_folds):
            opts = Options()
            opts.save_val_results = True
            opts.ckpt =Path(self.root /f"{fold}/best/best.pth")
            model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)

            model.eval()
            metrics = StreamSegMetrics(opts.num_classes)

            test_frames = [Path("dataset/test/images") / name for name in   os.listdir("dataset/test/images")]
            test_dst  = Mydataset(test_frames,preload = True)
            test_loader = data.DataLoader(test_dst, batch_size=20, shuffle=False,
            drop_last=True, pin_memory=True) 

            val_score, _ = validate(opts, model, test_loader, device, metrics, Path((self.root /f"{fold}/best"))) 
            scores[fold] = val_score
        
        with open(self.root / "test_score.json", "w") as f:
            json.dump(scores, f, indent= 4)




    def plot_optimization(self, fold: int):

        trials = self.studies[fold]
        completed_trials = trials[trials['state'] == 'COMPLETE']
        values = completed_trials['value']
        x = np.arange(len(values))
        
        plt.plot(completed_trials['number'], values)
        plt.show()

    def plot_training(self, metrics):
        
        train_loss = [m["loss"] for m in metrics['train']]
        train_x = [m["epoch"] for m in metrics['train']]

        for metric in self.metrics:
            eval_metric = [m[metric] for m in metrics['validation']]
            eval_x =  [m['epoch'] for m in metrics['validation']]
            plt.plot(eval_x, eval_metric, label= metric)

        #plt.plot(train_x, train_loss)
        plt.legend()
        plt.show()
    
    def plot_training_class_IoU(self, metrics):

        fig, ax = plt.subplots()

        train_loss = [m["loss"] for m in metrics['train']]
        train_x = [m["epoch"] for m in metrics['train']]

        bg_metric = [m["Class IoU"]['0'] for m in metrics['validation']]
        cow_metric = [m["Class IoU"]['1'] for m in metrics['validation']]
        heu_metric = [m["Class IoU"]['2'] for m in metrics['validation']]

        eval_x =  [m['epoch'] for m in metrics['validation']]
        ax.plot(eval_x, bg_metric, label= "background IoU")
        ax.plot(eval_x, cow_metric, label= "cowIoU")
        ax.plot(eval_x, heu_metric, label= "heu IoU")

        #plt.plot(train_x, train_loss)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        plt.show()
    
    def plot_training_best_model(self, fold):
        
        with open(self.root / f"{fold}"/ "best/metrics.json", 'r') as f:
            metrics = json.load(f)

        self.plot_training_class_IoU(metrics)

    def plot_test_metrics(self):

        cls_to_name = {'0' : "Hintergrund", '1': "Kuh", '2': "Sillage"}

        with open(self.root / "test_score.json", "r") as f:
            test_results = json.load(f)

        rows_test, rows_val = [], []

        cmap = plt.get_cmap("Set2")
        colors = cmap.colors

        for fold, test_metrics in test_results.items():

            with open(self.root / fold / "best/metrics.json", "r") as f:
                val_results = json.load(f)["validation"]

            test_flat = {k: v for k, v in test_metrics.items() if k != "Class IoU"}
            if "Class IoU" in test_metrics:
                for cls, v in test_metrics["Class IoU"].items():
                    test_flat[f"Class IoU {cls_to_name[cls]}"] = v
            rows_test.append(test_flat)

            best_epoch = max(val_results, key=lambda m: m["Mean IoU"])
            val_flat = {k: v for k, v in best_epoch.items()
                        if k not in ["epoch", "itr", "Class IoU"]}
            if "Class IoU" in best_epoch:
                for cls, v in best_epoch["Class IoU"].items():
                    val_flat[f"Class IoU {cls_to_name[cls]}"] = v

            rows_val.append(val_flat)
        df_test, df_val = pd.DataFrame(rows_test), pd.DataFrame(rows_val)

        # mean ± std
        mean_test, std_test = df_test.mean(), df_test.std()
        mean_val, std_val   = df_val.mean(), df_val.std()

        metrics = mean_test.index.tolist()
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, mean_test, width, yerr=std_test, capsize=3, label="Test", color= colors[0])
        ax.bar(x + width/2, mean_val,  width, yerr=std_val,  capsize=3, label="Validation", color= colors[4])

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Best Validation vs Test Metrics ")
        ax.set_ylim(0.7, 1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


    def plot_hyperparemter_optimization(self):

        df_all = pd.concat(self.studies, keys=range(len(self.studies)), names=["study_id", "row"])
        df_all = df_all[df_all["state"] == "COMPLETE"]
        agg = df_all.groupby("number")["value"].agg(["mean", "std"]).reset_index()

        plt.figure(figsize=(8,5))
        plt.plot(agg["number"], agg["mean"], label="Durchschnitt")
        plt.fill_between(agg["number"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.2, label="Standart Abweichung")
        plt.axvline(25, color="green", linestyle=":", label="Ende der Exploration")
        plt.xlabel("Anzahl Versuche")
        plt.ylabel("Mean IoU")
        plt.ylim(0, 0.2)
        plt.title("Hyerparameter Optimierungsverlauf")
        plt.legend()
        plt.show()

    def plot_lr(self):
        df_all = pd.concat(self.studies, keys=range(len(self.studies)), names=["study_id", "row"])
        df_all = df_all[df_all["state"] == "COMPLETE"]

        losses = ["focal_loss", "cross_entropy"]
        i = 0
        fig, ax = plt.subplots(len(losses))
        for loss in losses:
            df_lr = df_all[["value", "params_learning_rate", "params_loss"]]

            df_lr = df_lr[df_lr["params_loss"] == loss]
            bins = pd.qcut(df_lr["params_learning_rate"], 30)
            agg = df_lr.groupby(bins)["value"].mean()
            agg_var = df_lr.groupby(bins)["value"].std()
            intervals = pd.IntervalIndex(agg.index.categories)
            x = intervals.mid

            ax[i].plot(x, agg.values, label="mean IoU", linewidth = 0.8)

            ax[i].fill_between(x,
                            agg.values - agg_var.values,
                            agg.values + agg_var.values,
                            alpha=0.2, label="Standart Abweichung")

            ax[i].scatter(df_lr["params_learning_rate"], 
                        [ax[i].get_ylim()[0]]*(len(df_lr)) , marker = "|", s=10, color="black", alpha=0.2, label="Samples")
            ax[i].set_xscale('log')
            ax[i].set_title(f"Search Range Loss {loss}")
            ax[i].legend()
            i+=1

        plt.show()

    def plot_wd(self):

        df_all = pd.concat(self.studies, keys=range(len(self.studies)), names=["study_id", "row"])
        #df_all = df_all[df_all["state"] == "COMPLETE"]
        
        losses = ["focal_loss", "cross_entropy"]

        i = 0
        fig, ax = plt.subplots(len(losses))
        for loss in losses:
            df_lr = df_all[["value", "params_weight_decay", "params_loss"]]

            df_lr = df_lr[df_lr["params_loss"] == loss]
            bins = pd.qcut(df_lr["params_weight_decay"], 40)
            agg = df_lr.groupby(bins)["value"].mean()
            agg_var = df_lr.groupby(bins)["value"].std()
            intervals = pd.IntervalIndex(agg.index.categories)
            x = intervals.mid

            ax[i].plot(x, agg.values, label="mean IoU", linewidth = 0.8)

            ax[i].fill_between(x,
                            agg.values - agg_var.values,
                            agg.values + agg_var.values,
                            alpha=0.2, label="Standart Abweichung")

            ax[i].scatter(df_lr["params_weight_decay"], 
                        [ax[i].get_ylim()[0]]*(len(df_lr)) , marker = "|", s=10, color="black", alpha=0.2, label="Samples")
            ax[i].set_xscale('log')
            ax[i].set_title(f"Search Range Loss {loss}")
            ax[i].legend()
            i+=1

        plt.show()
    
    def plot_bg(self):

        df_all = pd.concat(self.studies, keys=range(len(self.studies)), names=["study_id", "row"])
        #df_all = df_all[df_all["state"] == "COMPLETE"]
        fig, ax = plt.subplots()

        df_lr = df_all[["value", "params_bg_weight"]]
        bins = pd.qcut(df_lr["params_bg_weight"], 50)
        agg = df_lr.groupby(bins)["value"].mean()
        agg_var = df_lr.groupby(bins)["value"].std()
        intervals = pd.IntervalIndex(agg.index.categories)
        x = intervals.mid

        ax.plot(x, agg.values, label="mean performance", linewidth = 0.8)

        ax.fill_between(x,
                        agg.values - agg_var.values,
                        agg.values + agg_var.values,
                        alpha=0.2, label="Standart Abweichung")

        ax.scatter(df_lr["params_bg_weight"], 
                    [ax.get_ylim()[0]]*(len(df_lr)) , marker = "|", s=5, color="black", alpha=0.1, label="sampled values")
        plt.show()

    def plot_parameter_importance(self):
        
        records = []
        for fold in range(self.num_folds):
            study = optuna.load_study(study_name=f"fold_optimization",
                                    storage=f"sqlite:///dataset/folds/{fold}/optuna_study.db")
            imp = optuna.importance.get_param_importances(study)
            for p, v in imp.items():
                    records.append({"fold": fold, "param": p, "importance": float(v)})
        
        df = pd.DataFrame(records)
        pivot = df.pivot_table(index="param", columns="fold", values="importance", aggfunc="mean")

        mean_imp = pivot.mean(axis=1, skipna=True)
        std_imp  = pivot.std(axis=1, ddof=1, skipna=True)
        count    = pivot.count(axis=1)

        order = mean_imp.sort_values(ascending=True).index 
        pivot = pivot.loc[order]
        mean_imp = mean_imp.loc[order]
        std_imp = std_imp.loc[order]
        count = count.loc[order]

        plt.figure(figsize=(7, max(2, 0.35 * len(mean_imp))))
        y = np.arange(len(mean_imp))
        plt.barh(y, mean_imp.values, xerr=std_imp.values, align="center", alpha=0.9, capsize=3)
        plt.yticks(y, mean_imp.index)
        plt.xlabel("Importance (mean across folds)")
        plt.title("Hyperparameter Importance Across Folds (mean ± std)")

        for yi, (m, s, n) in enumerate(zip(mean_imp.values, std_imp.values, count.values)):
            plt.text(m + (s if not np.isnan(s) else 0) + 0.01, yi, f"n={int(n)}", va="center")

        plt.tight_layout()
        plt.legend()
        plt.show()

        
if __name__ == "__main__":

    eval = Evaluation("c:/Users/Haufe/Desktop/dataset/folds")
    #eval.plot_parameter_importance()
    #eval.test_best_models()
    #eval.plot_test_metrics()
    eval.print_best_configs()



    





