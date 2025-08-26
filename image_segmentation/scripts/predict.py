import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lib.Deeplab.network as network
from util.options import Options
import argparse
from pathlib import Path
from PIL import Image

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import numpy as np


def load_model(ckpt_path: Path, device: torch.device):
    """
    Load a trained segmentation model from checkpoint.
    """
    opts = Options()
    opts.ckpt = ckpt_path
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, 
        output_stride=opts.output_stride
    )

    checkpoint = torch.load(str(opts.ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model


def run_inference(img_folder: Path, ckpt_path: Path, save_masks: bool, save_overlays: bool):
    """
    Run inference on a folder of imagees with a trained model.
    """
    if save_masks:
        mask_folder = Path(img_folder) / "masks"
        mask_folder.mkdir(exist_ok=True)

    if save_overlays:
        overlay_folder = Path(img_folder)/ "overlays"
        overlay_folder.mkdir(exist_ok=True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    model.eval()

    valid_exts = {".png", ".jpg", ".jpeg"}
    img_list = [img for img in img_folder.iterdir() if img.suffix.lower() in valid_exts]

    for img_path in  img_list:
        img = Image.open(img_path)
        width, height = 640, 480
        transform = v2.ToImage()
        img_t = img.resize((width, height))
        input_t = transform(img_t).unsqueeze(0).to(device, dtype=torch.float32)

        output = model(input_t)
        preds = output.detach().max(dim=1)[1].cpu().numpy().astype(np.uint8)[0]

        if save_masks:
            mask_path = mask_folder/ (img_path.stem + ".npy")   
            np.save(mask_path, preds)

        colors = {
            1: (1, 0, 0, 0.4),  
            2: (0, 1, 0, 0.4), 
        }

        # Create RGBA overlay initialized as transparent
        overlay = np.zeros((*preds.shape, 4))

        for cls, color in colors.items():
            overlay[preds == cls] = color

        fig, ax = plt.subplots()

        ax.imshow(img_t)
        ax.imshow(overlay)
        ax.axis("off")  

        if save_overlays:
            overlay_path = overlay_folder / (img_path.stem + ".png") 
            plt.savefig(overlay_path,dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
        
        plt.close()
        
        #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("ckpt", type=Path, help="Path to model checkpoint (.pth).")
    parser.add_argument("image_folder", type=Path, help="Path to image folder.")
    parser.add_argument("--save_mask",type=bool, default=False, help="saves the mask to folder.")
    parser.add_argument("--save_overlays", type= bool, default=False, help="saves mask overlays to folder")

    args = parser.parse_args()

    run_inference(args.image_folder, args.ckpt, args.save_mask,args.save_overlays)
