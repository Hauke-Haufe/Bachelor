import torch.utils.data as data
from torchvision.transforms import v2
import numpy as np
from PIL import Image

from pathlib import Path

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

    def __init__(self, frames_path, preload = True):
        
        self.images = frames_path
        self.masks = []
        self.preload = preload

        for image in self.images:
            image_path = Path(image)
            *parent_paths, _, filename = image_path.parts
            self.masks.append(Path(*parent_paths)/  "masks"/ filename)

        if preload:
            self.preloaded_images = []
            self.preloaded_masks = []

            for i in range(len(self.images)):
                self.preloaded_images.append(Image.open(self.images[i]).convert('RGB'))
                self.preloaded_masks.append( Image.open(self.masks[i]))


        assert (len(self.images) == len(self.masks))
    
    def __getitem__(self, index):


        if self.preload:
            img = self.preloaded_images[index]
            target = self.preloaded_masks[index]
        else:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index])
       
        width, height = img.size
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), 
            #v2.RandomRotation(degrees=(-20, 20)),
            v2.RandomCrop((int(0.8 * height),  int(0.8 * width))),
            v2.ToImage(),
           ])  
        jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
     
        img, target = transform(img, target)
        img =  jitter(img)
        

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]