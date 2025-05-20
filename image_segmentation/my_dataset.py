import os
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from pathlib import Path
import matplotlib.pyplot as plt


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class Mydataset(data.Dataset):
    cmap = voc_cmap()

    def __init__(self, frames_path, eval = False):
        

        self.images = frames_path
        self.masks = []
        
        if  eval:
            self.debug = True
        else:
            self.debug = False

        for image in self.images:
            image_path = Path(image)
            *parent_paths, _, filename = image_path.parts
            self.masks.append(Path(*parent_paths)/  "masks"/ filename)

        assert (len(self.images) == len(self.masks))
    
    def __getitem__(self, index):

        image_path = self.images[index]
        masks_path = self.masks[index]
        img = Image.open(self.images[index]).convert('RGB')

        """image = np.array(img.getdata()).reshape(img.size[1],img.size[0], 3)
        plt.imshow(image)
        plt.show()"""

        

        target = Image.open(self.masks[index])
        if self.debug:
            plt.imshow( np.array(target.getdata()).reshape(img.size[1],img.size[0], 1))
            plt.show()

        width, height = img.size
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), 
            #v2.RandomRotation(degrees=(-20, 20)),
            v2.RandomCrop((int(0.8 * height),  int(0.8 * width))),
            v2.Resize((int(0.6 * height), int(0.6 * width))),
            #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.ToImage(),
           ])
        
        img, target = transform(img, target)
        
        if self.debug:
            image = img.detach().cpu().numpy()
            image = image.transpose(1, 2, 0).astype(np.uint8)
            #plt.imshow(image)
            #plt.show()
            trg = target.detach().cpu().numpy()
            plt.imshow(trg.transpose(1,2,0))
            plt.show()


        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]