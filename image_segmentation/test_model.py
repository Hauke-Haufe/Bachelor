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
    opts.ckpt ="dataset/test/3_16_0.30000001192092896/checkpoints/best_deeplabv3plus_resnet50_Cow_segmentation_os16.pth"
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)

    return model

opts = Options()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model()

model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("data/model_scripted.pt")

model.train()
images = [file for file in  os.listdir("data/data_set/run3") if file.endswith(".png")]
width, height =640, 480
transform = transforms.Compose([transforms.ToTensor()])

with torch.no_grad():
    for image in images:
        
        img = Image.open(os.path.join("data/data_set/run3", image)).convert('RGB')

        image = transform(img).unsqueeze(0)
        e = torch.concat((transform(img).unsqueeze(0), transform(img).unsqueeze(0)))
        output = model(e)
        map =  output.detach().cpu().numpy()[0][1]
        plt.imshow(map, cmap = "hot")
        plt.colorbar()
        plt.show()
        plt.imshow(np.transpose(image.squeeze(0).detach().cpu().numpy(), (1, 2, 0)))
        preds =  output.detach().max(dim=1)[1].cpu().numpy()
        plt.imshow(preds[0], alpha=0.8)
        plt.show()
        
        #plt.imshow(output.squeeze(1).detach().cpu().numpy()[0][2])
        #plt.show()
    
