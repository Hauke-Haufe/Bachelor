from pathlib import Path
import os 
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

class Evalutation:

    def __init__(self, crossvalidation_root):

        self.root =crossvalidation_root
        self.eval_metrics = {"Overall Acc": 1 ,"Mean Acc": 1,"FreqW Acc": 1, "Mean IoU": 1, "Class IoU":1}
        
        
    def evaluate_configs(self, fold_path):

        fold_path = Path(fold_path)

        with open(fold_path / "index.json", "r") as f:
            index = json.load(f)

        metric_dict = {}
        for m in self.eval_metrics.keys():
            metric_dict[m] = {}

        for test in  index:
            test = Path(test)
            with open(test/"metrics.json", "r") as f:
                metrics = json.load(f)

            for m in self.eval_metrics.keys():
                if m =="Class IoU":
                    metric_dict[m][str(test)] = max(metrics["validation"], key=lambda d: d[m]["1"])
                else:
                    metric_dict[m][str(test)]  = max(metrics["validation"], key=lambda d: d[m])
            

        for m in self.eval_metrics.keys():
            if m =="Class IoU":
                metric_dict[m] = OrderedDict(sorted(metric_dict[m].items(), key=lambda item: item[1][m]["1"], reverse=True))
            else:
                metric_dict[m] = OrderedDict(sorted(metric_dict[m].items(), key=lambda item: item[1][m], reverse=True))

        
        return metric_dict    

    def determine_best_test(self, fold_path):
        
        evaluation = self.evaluate_configs(fold_path)
        fold_path = Path(fold_path)

        with open(fold_path / "index.json", "r") as f:
            index = json.load(f)

        metric_rank_dict = { config: {m: list(evaluation[m]).index(config) 
                                      for m in self.eval_metrics.keys()} 
                            for config in index
        }
        
        overall_score = {
            config: sum( weight * metric_rank_dict[config][m] for m, weight in self.eval_metrics.items())
            for config in index
        }

        overall_score = dict(sorted(overall_score.items(), key=lambda item: item[1]))
        
        return overall_score

    def plot_train_loss(self, ax, metrics):

        loss = [ loss["loss"]  for loss in metrics["train"]]
        epoch = [ loss["epoch"]  for loss in metrics["train"]]

        ax.plot(np.array(epoch), np.array(loss), label = "loss")
        return ax

    def plot_class_ious(self, ax, metrics):

        classes = [ label for label, _ in metrics["validation"][0]["Class IoU"].items()]
        support = np.array([ value["epoch"]  for value in metrics["validation"]])

        for label in classes:
            loss = [ loss["Class IoU"][label] for loss in metrics["validation"]]
            ax.plot(support, np.array(loss), label = label)
        
        return ax

    def plot_traning(self, metrics):

        fig = plt.figure()
        ax = fig.add_subplot()

        p = Path("dataset/test/bt=13_str=16_cw=0.4_lr=0.001_wd=0.01")

        with open(p /"metrics.json", "r") as f:
            metrics = json.load(f)

        ax = self.plot_train_loss(ax, metrics)
        ax = self.plot_class_ious(ax, metrics)
        
        ax.legend()
        plt.show()
        


if __name__ == "__main__":

    eval = Evalutation("dataset/folds")
    #eval.evaluate_configs("dataset/folds/0")
    best = eval.determine_best_test("dataset/folds/0")


        



