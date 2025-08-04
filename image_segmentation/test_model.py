import os
import os
import numpy as np
import lib.Deeplab.network as network


import torch
import torch.nn as nn
from cross_validation import Options
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path

def load_model():  

    opts = Options()
<<<<<<< HEAD
    opts.ckpt =Path("dataset/test/bt=4_str=8_cw=0.1444_wd=0.000446_lr=0.005609_l=focal_loss_lp=poly_fb=True/best.pth")
=======
    opts.ckpt =Path("dataset/folds/0/bt=7_str=16_cw=0.3_lr=0.01_wd=0.01_l=cross_entropy/best.pth")
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)

    return model

run_path = "data/data_set/run5"

opts = Options()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model()

<<<<<<< HEAD
batch = 10

=======
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
model.train()
images = [file for file in  os.listdir(run_path) if file.endswith(".png")]
width, height =640, 480
transform = transforms.Compose([transforms.ToTensor()])

with torch.no_grad():
<<<<<<< HEAD
    for i in range(0, len(images)- batch +1, batch):
        
        img = Image.open(os.path.join(run_path, images[i])).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        for j in range(1, batch):
            img = Image.open(os.path.join(run_path, images[i])).convert('RGB')
            tensor = torch.concat((tensor, transform(img).unsqueeze(0)))

        output = model(tensor)
=======
    for image in images:
        
        img = Image.open(os.path.join(run_path, image)).convert('RGB')

        image = transform(img).unsqueeze(0)
        e = torch.concat((transform(img).unsqueeze(0), transform(img).unsqueeze(0)))
        output = model(e)
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
        map =  output.detach().cpu().numpy()[0][1]
        plt.imshow(map, cmap = "hot")
        plt.colorbar()
        plt.show()
<<<<<<< HEAD
        plt.imshow(np.transpose(tensor[0].squeeze(0).detach().cpu().numpy(), (1, 2, 0)))
=======
        plt.imshow(np.transpose(image.squeeze(0).detach().cpu().numpy(), (1, 2, 0)))
>>>>>>> 1dd28801958d71b51b3a4ffad52ebb0842f3d18f
        preds =  output.detach().max(dim=1)[1].cpu().numpy()
        plt.imshow(preds[0], alpha=0.8)
        plt.show()
        
        #plt.imshow(output.squeeze(1).detach().cpu().numpy()[0][2])
        #plt.show()
    
