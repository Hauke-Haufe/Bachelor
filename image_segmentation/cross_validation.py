import random
import os 
from pathlib import Path
import torch
import copy
import itertools
import json
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
        self.save_val_results = True

        #constants
        self.val_interval = 50
        self.total_itrs = 2000
        self.val_batch_size = 10
        self.loss_type = 'cross_entropy'

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

    batch_vals = [3, 7, 13]
    stride = [8,16]
    background_weighting = [0.05, 0.15, 0.3, 0.5]

    
    total_com = len(batch_vals)*len(stride)*len(background_weighting)
    
    to_remove_total = 23

    folds_path = Path(folds_path)
    for path in folds_path.iterdir():
         
        if not (path/ "grid.json" ).is_file():

            grid = list(itertools.product(*[batch_vals, stride, background_weighting]))
            grid = [list(item) for item in grid]


            with open(path/  "grid.json", "w") as file:
                json.dump(grid, file , indent= 4)
        

        with open(path/ "grid.json", "r") as file:
            grid = json.load(file)
        
        already_removed = total_com - len(grid)
        num_picks = max(0, to_remove_total - already_removed)
        hyperparameter_optimization(grid, num_picks, path)

def hyperparameter_optimization(grid, num_picks, fold_path):

    for i in range(num_picks):
        opts = Options()

        index = random.randrange(len(grid))
        hyperparameters = grid[index]
        opts.batch_size = hyperparameters[0]
        opts.output_stride = hyperparameters[1]
        opts.class_weights[0] = hyperparameters[2]

        config_path = fold_path/ f"{opts.batch_size}_{opts.output_stride}_{opts.class_weights[0]}"
        if config_path.is_dir() and (fold_path / f'latest_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth').is_file():

            checkpoint_path = fold_path / f'latest_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth'
            checkpoint = torch.load(checkpoint_path , map_location=torch.device('cpu'), weights_only=False)

            if checkpoint["cur_itrs"] <= opts.total_itrs:
                opts.ckpt = checkpoint_path
                opts.continue_training = True
                train(opts, config_path)

        else:
            (config_path).mkdir(parents=True, exist_ok=True)
            train(opts, config_path)

        grid.pop(index)
        
        with open(fold_path/  "grid.json", "w") as file:
            json.dump(grid, file , indent= 4)
    
    print()

def test():
    opts = Options()
    opts.batch_size = 3
    opts.output_stride = 16
    opts.class_weights[0] = 0.1
    opts.class_weights[2] = 1.5
    opts.total_itrs = 10000
    opts.val_interval = 100
    opts.val_batch_size = 10

    path = Path("dataset")/ "test"/  f"{opts.batch_size}_{opts.output_stride}_{opts.class_weights[0]}"

    train(opts, path)

def result():

    fold_paths = Path("dataset/folds")
    results = {}

    for fold in fold_paths.iterdir():

        for config in fold.iterdir():
            checkpoint_path = config / 'checkpoints'

            if (checkpoint_path).is_dir():
                result_path = [file for file in os.listdir( checkpoint_path) if file.split("_")[0] == "best"]
                if len(result_path) == 1:
                    model_path = checkpoint_path/ result_path[0]
                    result =  torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                    hyp = config.parts[-1]
                    results[hyp] = (result["best_score"], result["cur_itrs"], fold.parts[-1], model_path)

    sorted_results = sorted(results.items(), key = lambda item: item[1][0], reverse = True)
    print(sorted_results)
    """    best = sorted_results[0]
    stride = best[0].split("_")[-1]

    opts = Options()
    opts.output_stride = stride
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    checkpoint = torch.load( best[1][3], map_location=torch.device('cpu'), weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save("data/model_scripted.pt")"""
    

if __name__ == "__main__":
    #create_folds(5)
    #cross_validation("dataset/folds")
    #result()
    test()
