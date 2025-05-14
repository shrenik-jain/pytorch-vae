import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(DeepVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # (1, 256, 256) -> (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# -> (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# -> (256, 16, 16)
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # (256, 16, 16) -> (128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> (64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> (32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # -> (1, 256, 256)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def is_valid_file(path):
    return path.lower().endswith(('.jpg', '.jpeg'))
# Dataset and loader (Assume data is in ./ct_data/)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = ImageFolder(root='/content/chest_xray/chest_xray', transform=transform,is_valid_file=is_valid_file)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom ImageFolder that skips hidden folders
class CleanImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def is_valid_file(path):
    return path.lower().endswith(('.jpg', '.jpeg'))

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CleanImageFolder(root='/content/chest_xray/chest_xray', transform=transform, is_valid_file=is_valid_file)
loader = DataLoader(dataset, batch_size=32, shuffle=True)



# Model
vae = DeepVAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
e_loss = []
# Training loop
epochs = 100
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {train_loss / len(loader.dataset):.4f}")
    e_loss.append(train_loss / len(loader.dataset))
    
    if epoch%25 == 0:
        # Save sample reconstructions
        vae.eval()
        with torch.no_grad():
            z = torch.randn(64, vae.latent_dim).to(device)
            samples = vae.decode(z)
            vutils.save_image(samples, f"/content/v256_samples_epoch{epoch+1}.png", nrow=8, normalize=True)

torch.save({
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': e_loss,
}, '/content/vae_256_trained_final.pth')