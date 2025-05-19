import random
import os 
from pathlib import Path
import torch
import copy
import itertools
import json
from train import train

class Options():

    def __init__(self):
        #model
        self.model = "deeplabv3plus_mobilenet"
        self.num_classes = 3
        self.output_stride = 8

        self.dataset = "Cow_segmentation"
        self.save_val_results = True

        #constants
        self.val_interval = 50
        self.total_itrs = 2000
        self.val_batch_size = 10
        self.loss_type = 'cross_entropy'

        #hyperparameter
        self.batch_size = 15

        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001

        #visualize Option
        self.enable_vis = None

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.ckpt= None #"data/checkpoints/best_deeplabv3plus_mobilenet_Cow_segmentation_os8.pth" #"data\checkpoints\latest_deeplabv3plus_resnet50_Cow_segmentation_os8.pth" 
        self.continue_training = False

def create_folds(k_folds):

    run_path = Path("dataset/runs")
    folds_path = Path("dataset/folds")
   
    folds = []

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
            train_frames =frames[:val_start] +frames[val_end:]

            train_paths.extend([run /"images"/ train_frame for train_frame in train_frames])
            eval_paths.extend([ run /"images"/ val_frame for val_frame in val_frames])
        
        with open(fold_path / "train.txt", "w") as f:
            for path in train_paths:
                f.write(f"{path}\n")

        with open(fold_path / "eval.txt" , "w") as f:
            for path in eval_paths:
                f.write(f"{path}\n")

    return folds

def cross_validation(folds_path):

    max_count_com = 6 #todo das hier ist nbocht richtig  
    batch_vals = [3, 7, 13]
    stride = [8,16]
    total_com = len(batch_vals)*len(stride)

    folds_path = Path(folds_path)
    for path in folds_path.iterdir():
         
        if not (path/ "grid.json" ).is_file():

            grid = list(itertools.product(*[batch_vals, stride]))
            grid = [list(item) for item in grid]

            with open(path/  "grid.json", "w") as file:
                json.dump(grid, file , indent= 4)
        

        with open(path/ "grid.json", "r") as file:
            grid = json.load(file)
        

        if total_com - len(grid) < max_count_com:
            num_picks =  max_count_com - total_com + len(grid) 
            hyperparameter_optimization(grid, num_picks, path)


def hyperparameter_optimization(grid, num_picks, fold_path):

    for i in range(num_picks):
        opts = Options()

        index = random.randrange(len(grid))
        hyperparameters = grid[index]
        opts.batch_size = hyperparameters[0]
        opts.output_stride = hyperparameters[1]

        if (fold_path/ f"{opts.batch_size}_{opts.output_stride}").is_dir() and (fold_path / f'latest_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth').is_file():
            checkpoint_path = fold_path / f'latest_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth'
            checkpoint = torch.load(checkpoint_path , map_location=torch.device('cpu'), weights_only=False)

            if checkpoint["cur_itrs"] <= opts.total_itrs:
                opts.ckpt = checkpoint_path
                opts.continue_training = True
                train(opts,fold_path/ f"{opts.batch_size}_{opts.output_stride}")

        else:
            (fold_path/ f"{opts.batch_size}_{opts.output_stride}").mkdir(parents=True, exist_ok=True)
            train(opts, fold_path/ f"{opts.batch_size}_{opts.output_stride}")

        grid.pop(index)
        
        with open(fold_path/  "grid.json", "w") as file:
            json.dump(grid, file , indent= 4)

if __name__ == "__main__":
    #create_folds(5)
    cross_validation("dataset/folds")