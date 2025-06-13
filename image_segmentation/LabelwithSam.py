from PIL import Image, ImageTk
import time

from pathlib import Path

import numpy as np
import torch
import numpy as np
import torch.nn.functional as F

import tkinter as tk
from tkinter import filedialog
"""from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog"""
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def convert_mask_to_sam_logits(mask_binary, target_size=(256, 256), fg_value=10.0, bg_value=-10.0):
    """
    Convert binary mask to SAM-compatible mask_input (logits format).
    
    Parameters:
    - mask_binary: numpy array or tensor of shape (H, W), values 0 or 1
    - target_size: size expected by SAM (e.g., H//4, W//4 of input image)
    - fg_value: logit value for foreground
    - bg_value: logit value for background

    Returns:
    - mask_input: torch.Tensor of shape [1, 1, H', W'] with logit values
    """
    if isinstance(mask_binary, np.ndarray):
        mask_binary = torch.tensor(mask_binary, dtype=torch.float32)

    # Ensure shape [1, 1, H, W]
    mask_tensor = mask_binary[None, None, :, :]  # shape [1, 1, H, W]

    # Resize to match SAM encoder resolution (typically H//4, W//4)
    mask_resized = F.interpolate(mask_tensor, size=target_size, mode="bilinear", align_corners=False)

    # Convert to logits: fg_value for 1s, bg_value for 0s
    mask_logits = mask_resized * (fg_value - bg_value) + bg_value  # linear map from 0→bg to 1→fg

    return mask_logits[0,0]

class LabelwithSam:

    def __init__(self, sam, image):

        self.root =  tk.Tk() 

        #--------------------------------------
        #----------Setup Window Layout --------
        #--------------------------------------
        self.root.title("Label with Sam")
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=400, height=300, bg="white")
        self.canvas.pack()

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)


        #--------------------------------------
        #---------set start Parameters---------
        #--------------------------------------

        #self.sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
        self.sam = sam_model_registry["vit_b"](checkpoint="data/sam_vit_b.pth")
        #self.sam = sam_model_registry["vit_l"](checkpoint="data/sam_vit_l.pth")
        self.sam.to("cuda")
        self.sam = SamPredictor(self.sam)

        self.masks = [] 
        self.points = []
        self.img = Image.open(image)

        self.p_flag = True
        self.vis = True


        self.classes = {0:{"name": "object", "color": (255, 0, 0, 100)},
                        1: {"name": "cow", "color": (255,192,203, 100)}, 
                        2:{"name":"heu", "color": (255, 255, 0, 100)}}

        #--------------------------------------
        #-----------Point Keybinds-------------
        #--------------------------------------
        # Binding mouse click event
        self.canvas.bind("<Button-1>", self.create_point)
        # Switch Point insert mode
        self.canvas.bind("<Button-3>", self.change_input_mode)
        #delets the last point
        self.root.bind("<KeyPress-d>", self.delete_last_point)
        #delete all Points
        self.root.bind("<KeyPress-q>", self.clear_all_points)


        #--------------------------------------
        #-----------Mask Keybinds-------------
        #--------------------------------------
        #select/unselect mask
        self.canvas.bind("<Control-Button-1>", self.select_mask)
        #delets selected masks
        self.root.bind("<Control-KeyPress-d>", self.delete_selected_masks)
        # generate mask from seleted prompts 
        self.root.bind("<KeyPress-g>", self.generate_mask)
        
        for i, _ in self.classes.items():
            self.root.bind(str(i), self.on_number)

        #zooming 
        self.zoom_factor = 1.0
        self.zoom_step = 0.1  # step for each scroll
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Control-KeyPress-Up>", self.zoom_in)
        self.root.bind("<Control-KeyPress-Down>", self.zoom_out)
        
        #reset for new mask creation
        self.root.bind("<KeyPress-r>", self.reset)

        #toggle vis Elements
        self.root.bind("<KeyPress-v>", self.toggle_vis)

        #next Image button
        next_btn = tk.Button(control_frame, text="Finish", command=self.finish_image)
        next_btn.pack(side="left", pady=2)

        #self.init_segment()
        
        self.render()
        self.root.mainloop()


    def init_segment(self):

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Confidence threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        predictor = DefaultPredictor(cfg)
        outputs = predictor(np.array(self.img))

        for mask in outputs["instances"].pred_masks.cpu().numpy():
            logit = convert_mask_to_sam_logits(mask)
            self.masks.append({"mask":mask, "selected": False, "logits": logit})
    
    #--------------------------------------
    #gui functionality
    #--------------------------------------
    def on_mousewheel(self, event):
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
        elif event.delta < 0 or event.num == 5:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)

        self.render()

    def on_number(self, event):

        number_pressed = event.char
        if number_pressed.isdigit():
            for i in range(len(self.masks)):
                if self.masks[i]["selected"]:
                    self.masks[i]["class"] = int(number_pressed)

        self.reset(None)
        self.render()
    
    def zoom_in(self, event):

        self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
        self.render()

    def zoom_out(self, event):

        self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
        self.render()

    def reset(self, event):

        for i in range(len(self.masks)):
            if self.masks[i]["selected"]:
                self.masks[i]["selected"] = False
        
        self.clear_all_points(None)
        self.p_flag = True
        self.render()
    
    def create_point(self, event):
        x, y = event.x, event.y
        x = int(x/self.zoom_factor)
        y = int(y/self.zoom_factor)

        radius = 3
        if self.p_flag:
            self.points.append({"coord": (x, y), "inside":True})
        else:
            self.points.append({"coord": (x, y), "inside":False})

        self.render()
    
    def change_input_mode(self, event):
        self.p_flag = not self.p_flag

    def clear_all_points(self, event):
        self.points.clear()

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def delete_last_point(self, event):

        if len(self.points)> 0:
            self.points.pop(-1)
            self.render()

    def select_mask(self, event):
        x, y = event.x, event.y
        x = int(x/self.zoom_factor)
        y = int(y/self.zoom_factor)
        for i in range(len(self.masks)):
                if self.masks[i]["mask"][y,x]:
                    self.masks[i]["selected"] = not self.masks[i]["selected"]

        self.render()

    def delete_selected_masks(self,event):

        self.masks = [mask for mask in self.masks if not mask["selected"]]

        self.render()
    
    def toggle_vis(self, event):
        self.vis = not self.vis
        self.render()

    def finish_image(self):

        self.root.quit()
        self.root.destroy()
        
    def get_mask(self):
        
        return [
            {"mask": mask["mask"], 
            "class" : self.classes[mask["class"]]["name"]} 
            for mask in self.masks]

    def generate_mask(self, event):
        
        self.sam.set_image(np.array(self.img))
        
        labels, points = [], []
        for point in self.points:
            
            points.append(point["coord"])
            labels.append(point["inside"])

        if len(points) >0:
            points = np.asanyarray(points)
            labels = np.asanyarray(labels)
        else:
            points = None
            labels = None
        
        mask_in = self.get_selected_mask()
        if mask_in is not None:
            mask_in = mask_in
            
        for i in range(4):
            sam_masks, scores, logits = self.sam.predict(
                    point_coords=points,
                    point_labels = labels,
                    mask_input= mask_in,
                    multimask_output=False,
            )
            mask_in = logits[np.argmax(scores), :, :]
            mask_in = mask_in[None, :, :]

        self.masks = [mask for mask in self.masks if not mask["selected"]]
        self.masks.append({"mask":sam_masks[0], "selected": True, "logits": logits[np.argmax(scores), :, :], "class": 0})
        self.render()

    def get_selected_mask(self):

        selected_masks = []
        for mask in self.masks:

            if mask["selected"]:
                selected_masks.append(mask["logits"])

        if len(selected_masks) > 0:
            return np.asanyarray(selected_masks)
        else:
            return None
    
    #--------------------------------------
    #render Funktion
    #--------------------------------------
    def render(self):
        
        
        new_size = (int(self.img.width * self.zoom_factor),
                    int(self.img.height * self.zoom_factor))

        if self.vis:
            img_overlay = self.img.convert("RGBA")
            self.tk_img = ImageTk.PhotoImage(self.img, master = self.root)

            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

            for mask in self.masks:
                overlay = Image.new("RGBA", self.img.size, (0, 0, 0, 0))

                # Define overlay color (e.g., red) and alpha (transparency)
                if mask["selected"]:
                    overlay_color = (0, 255, 0, 100)
                else:
                    overlay_color = self.classes[mask["class"]]["color"]

                # Convert the binary mask to a uint8 image with RGBA color where mask==1
                # mask must be shape (H, W)
                mask_rgba = np.zeros((*mask["mask"].shape, 4), dtype=np.uint8)
                mask_rgba[mask["mask"] == 1] = overlay_color

                # Create a PIL image from the RGBA array
                mask_img = Image.fromarray(mask_rgba, mode="RGBA")
                img_overlay = Image.alpha_composite(img_overlay, mask_img)
            
            self.tk_img = ImageTk.PhotoImage(img_overlay.resize(new_size, Image.LANCZOS))

            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

            radius = 3
            for point in self.points:
                x, y = point["coord"]
                zx, zy = x * self.zoom_factor, y * self.zoom_factor
                if point["inside"]:
                    self.canvas.create_oval(zx-radius,zy-radius, zx+radius, zy+radius, fill="green")
                else:
                    self.canvas.create_oval(zx-radius,zy-radius, zx+radius, zy+radius, fill="red")

        else:
            self.tk_img = ImageTk.PhotoImage(self.img.resize(new_size, Image.LANCZOS))
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

    
if __name__ == "__main__":

    #self.sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
    sam = sam_model_registry["vit_b"](checkpoint="data/sam_vit_b.pth")
    #self.sam = sam_model_registry["vit_l"](checkpoint="data/sam_vit_l.pth")
    sam.to("cuda")

    for image in Path("data/images/color").iterdir():
        label = LabelwithSam(sam, image)
        masks = label.get_mask()
        del label



