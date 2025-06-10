import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import time

class LabelwithSam:

    def __init__(self, run_folder):

        self.root = tk.Tk()

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

        info_field = tk.Entry(control_frame, width=20)
        info_field.pack(side=tk.LEFT, padx=5)

        #--------------------------------------
        #---------set start Parameters---------
        #--------------------------------------
        self.run_folder = Path(run_folder).iterdir()

        self.sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h.pth")
        self.sam.to("cuda")
        self.sam = SamPredictor(self.sam)

        self.img = Image.open(next(self.run_folder))
        self.masks = [] 
        self.points = []

        self.p_flag = True
        self.vis = True

        self.render()

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
        
        #toggle vis Elements
        self.root.bind("<KeyPress-v>", self.toggle_vis)


        #next Image button
        next_btn = tk.Button(control_frame, text="Next image", command=self.next_image)
        next_btn.pack(side="left", pady=2)

        self.root.mainloop()

    def create_point(self, event):
        x, y = event.x, event.y
   
        radius = 3
        if self.p_flag:
            self.points.append({"coord": (x, y), "inside":True})
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="green")
        else:
            self.points.append({"coord": (x, y), "inside":False})
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="red")
    
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
        for i in range(len(self.masks)):
                if self.masks[i]["mask"][y,x]:
                    self.masks[i]["selected"] = not self.masks[i]["selected"]

        self.render()

    def delete_selected_masks(self,event):
        for mask in self.masks:
            if mask["selected"]:
                self.masks.remove(mask)

        self.render()
    
    def toggle_vis(self, event):
        self.vis = not self.vis
        self.render()

    def next_image(self):

        self.masks = []
        self.points = []
        self.img = Image.open(next(self.run_folder))
        self.render()


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
            mask_in = mask_in[None, :, :]
            
        for i in range(2):
            sam_masks, scores, logits = self.sam.predict(
                    point_coords=points,
                    point_labels = labels,
                    mask_input= mask_in,
                    multimask_output=False,
            )
            mask_in = logits[np.argmax(scores), :, :]
            mask_in = mask_in[None, :, :]

        self.masks = [mask for mask in self.masks if not mask["selected"]]
        self.masks.append({"mask":sam_masks[0], "selected": True, "logits": logits[np.argmax(scores), :, :]})
        self.render()

    def get_selected_mask(self):

        selected_masks = []
        for mask in self.masks:

            if mask["selected"]:
                return(mask["logits"])

    def render(self):
        
        if self.vis:
            img_overlay = self.img.convert("RGBA")
            self.tk_img = ImageTk.PhotoImage(self.img)

            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

            for mask in self.masks:
                overlay = Image.new("RGBA", self.img.size, (0, 0, 0, 0))

                # Define overlay color (e.g., red) and alpha (transparency)
                if mask["selected"]:
                    overlay_color = (0, 255, 0, 100)
                else:
                    overlay_color = (255, 0, 0, 100)  # red with alpha=100/255

                # Convert the binary mask to a uint8 image with RGBA color where mask==1
                # mask must be shape (H, W)
                mask_rgba = np.zeros((*mask["mask"].shape, 4), dtype=np.uint8)
                mask_rgba[mask["mask"] == 1] = overlay_color

                # Create a PIL image from the RGBA array
                mask_img = Image.fromarray(mask_rgba, mode="RGBA")
                img_overlay = Image.alpha_composite(img_overlay, mask_img)
            
            self.tk_img = ImageTk.PhotoImage(img_overlay)

            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

            radius = 3
            for point in self.points:
                coord = point["coord"]
                if point["inside"]:
                    self.canvas.create_oval(coord[0]-radius,coord[1]-radius, coord[0]+radius, coord[1]+radius, fill="green")
                else:
                    self.canvas.create_oval(coord[0]-radius, coord[1]-radius, coord[0]+radius, coord[1]+radius, fill="red")

        else:
            self.tk_img = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())

if __name__ == "__main__":
    
    app = LabelwithSam("data/data_set/run2")
