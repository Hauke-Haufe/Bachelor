import os
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms


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

    def __init__(self,root):

        names = os.listdir(os.path.join(root, "images"))

        self.images = [os.path.join(root,"images", x) for x in names]
        self.masks = [os.path.join(root,"masks", x) for x in names]
        assert (len(self.images) == len(self.masks))


    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        transform = transforms.ToTensor()
        img = transform(img)
        target = transform(target)

        return img, target

    #hier noch was machen
    def transform(self, img, target):

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]