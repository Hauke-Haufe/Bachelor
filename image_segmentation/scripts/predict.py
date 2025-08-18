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


def run_inference(img_folder: Path, ckpt_path: Path):
    """
    Run inference on a folder of imagees with a trained model.
    """
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

        plt.imshow(img_t)
        plt.imshow(preds, alpha=0.5)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("ckpt", type=Path, help="Path to model checkpoint (.pth).")
    parser.add_argument("image_folder", type=Path, help="Path to image folder.")

    args = parser.parse_args()

    run_inference(args.image_folder, args.ckpt)
