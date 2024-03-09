import torch
import torch.nn as nn
import torch.nn.functional as F
from polip import decider

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as tr
from models import VQVAE
import argparse

device = decider("mps")

parser = argparse.ArgumentParser(description = "Generate an image")
parser.add_argument("--model_path", type = str, default = "model_weight/model_1.pt")
parser.add_argument("--device", type = str, default = "mps")
args = parser.parse_args()


device = torch.device(args.device)

vqvae = VQVAE(128, 32, 2, 512, 64, 0.25).to(device)
vqvae.load_state_dict(torch.load(args.model_path))

random_tensor = torch.randn(1, 3, 256, 256).to(device)

out = vqvae(random_tensor)


image = out[1]

image = image.squeeze(0)

transform = tr.Compose([tr.ToPILImage()])

img_pill = transform(image)
plt.imshow(img_pill)
plt.axis("off")
plt.show()