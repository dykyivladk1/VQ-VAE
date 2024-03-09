import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VQVAE
from tqdm import tqdm

from polip import decider, CustomImageDataset
import argparse
import torchvision.transforms as tr
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description = "Train a VQVAE")
parser.add_argument("--root_dir", type = str, default = "data/")
parser.add_argument("--batch_size", type = str, default = 32)
parser.add_argument("--epochs", type = int, default = 30)
parser.add_argument("--num_workers", type = int, default = 0)
parser.add_argument("--device", type = str, default = "mps")
args = parser.parse_args()

device = torch.device(args.device)

mean_std = tuple(0.5 for i in range(3))

transform = tr.Compose([
    tr.Resize((256, 256)),
    tr.ToTensor(),
    tr.Normalize(mean_std, mean_std)
])

ds = CustomImageDataset(image_dir = args.root_dir, transform=transform)

dl = torch.utils.data.DataLoader(ds, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers)

model = VQVAE(128, 32, 2, 512, 64, 0.25).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr = 3e-4, amsgrad = True)

model.train()

num_epochs = args.epochs


losses = []

if __name__ == "__main__":
    for epoch in range(num_epochs):
        with tqdm(enumerate(dl), total=len(dl)) as t:
            for idx, (image) in t:
                image = image.to(device)
                optimizer.zero_grad()
                embedding_loss, x_hat, perplexity = model(image)
                reconstruction_loss = F.mse_loss(x_hat, image)
                loss = reconstruction_loss + embedding_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if idx % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(dl)}], Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.4f}")
                    
            torch.save(model.state_dict(), f"model_weights/model_{epoch}.pt")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='blue')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()