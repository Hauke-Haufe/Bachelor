import os
from tqdm import tqdm
import lib.Deeplab.utils as utils
import os
import random
import numpy as np
import lib.Deeplab.network as network

from torch.utils import data
from lib.Deeplab.metrics import StreamSegMetrics

import torch
import torch.nn as nn
from lib.Deeplab.utils.visualizer import Visualizer
from PIL import Image
from my_dataset import Mydataset
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
import time


#todo p[ath in validate 
def get_dataset(path):
    
    *parent_paths, _ = path.parts
    parent_paths = Path(*parent_paths)

    with open(parent_paths / "train.txt", "r") as f:
        train_frames = f.read().splitlines()

    with open(Path(parent_paths) / "eval.txt", "r") as f:
        eval_frames = f.read().splitlines()

    train_dst  = Mydataset(train_frames,preload = True)
    val_dst  = Mydataset(eval_frames, preload=True)

    return train_dst, val_dst

def validate(opts, model, loader, device, metrics, path, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        result_path = path / "results"
        result_path.mkdir(parents=True, exist_ok=True)
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    target = targets[i]
                    pred = preds[i]

                    
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    #Image.fromarray(image,  mode='RGB').save(result_path /f"{img_id}_image.png")
                    #Image.fromarray(np.squeeze(target)).save(result_path /f"{img_id}_target.png")
                    #Image.fromarray(pred).save(result_path /f"{img_id}_pred.png")

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(result_path /f"{img_id}_overlay.png", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def train(opts, fold_path):
    
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                        env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    #get the Dataset
    train_dst, val_dst = get_dataset(fold_path)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=False,
        drop_last=True, pin_memory=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, pin_memory=True)

    # get Model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    layers_to_freeze = ["conv1", "layer1", "layer2", "layer3"]
    for name, module in model.backbone.named_children():
        if name in layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
    
    """for param in model.backbone.parameters():
        param.requires_grad = False"""

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # Set up Learning rate scheduler
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight= opts.class_weights)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):

        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  

        with open(fold_path/"metrics.json", "r") as f:
            t_metrics = json.load(f)   
        
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

        t_metrics = {"train": [], "validation": []}
        with open(fold_path/"metrics.json", "w") as f:
            json.dump(t_metrics, f)

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1

        for (images, labels) in train_loader:
                
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels.squeeze(1))

            loss.backward()
            optimizer.step()

            interval_loss += loss.item()

            if (cur_itrs) % 10 == 0:

                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                        (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                
                t_metrics["train"].append( {"epoch": cur_epochs, 
                                            "itr": cur_itrs, 
                                            "loss": interval_loss})
                
                with open(fold_path /"metrics.json", "w") as f:
                    json.dump(t_metrics, f, indent= 4)

                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:

                save_ckpt(fold_path /  f'latest.pth')
                print("validation...")
                model.eval()
                
                val_score, _ = validate(opts=opts, model=model, loader=val_loader, 
                    device=device, metrics=metrics, path=fold_path)
                print(metrics.to_str(val_score))


                t_metrics["validation"].append({
                    "epoch": cur_epochs, 
                    "itr": cur_itrs,
                    "Overall Acc": val_score["Overall Acc"],
                    "Mean Acc":val_score["Mean Acc"],
                    "FreqW Acc":val_score["FreqW Acc"],
                    "Mean IoU":val_score["Mean IoU"],
                    "Class IoU": val_score["Class IoU"],
                })

                with open(fold_path /"metrics.json", "w") as f:
                    json.dump(t_metrics, f,  indent= 4)

                # save best model
                if val_score['Mean IoU'] > best_score: 
                    
                    best_score = val_score['Mean IoU']
                    save_ckpt(fold_path /  f'best.pth')
                
                if best_score - val_score['Mean IoU'] > opts.max_decrease:
                    print("model Perfomence decrease too much")
                    return

                model.train()

            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

if __name__ == "__main__":
    pass