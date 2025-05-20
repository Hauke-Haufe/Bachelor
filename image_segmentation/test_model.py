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


def load_model():  

    opts = Options()
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)

    return model

opts = Options()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model()
#model.eval()
model.train()
images = [file for file in  os.listdir("data/data_set/run5") if file.endswith(".png")]
width, height =640, 480
transform = transforms.Compose([transforms.ToTensor()])

with torch.no_grad():
    for image in images:
        
        img = Image.open(os.path.join("data/data_set/run5", image)).convert('RGB')

        image = transform(img).unsqueeze(0)
        e = torch.concat((transform(img).unsqueeze(0), transform(img).unsqueeze(0)))
        
        plt.imshow(np.transpose(image.squeeze(0).detach().cpu().numpy(), (1, 2, 0)))
        plt.show()
        output = model(e)
        #plt.imshow(output.squeeze(1).detach().cpu().numpy()[0][2])
        #plt.show()
        preds =  output.detach().max(dim=1)[1].cpu().numpy()

        plt.imshow(preds[0])
        plt.show()
