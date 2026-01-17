import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BdSLDataset, MockDataset
from models.diffusion import Unet1D
import os
import argparse
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    if args.use_mock:
        dataset = MockDataset(num_samples=100)
    else:
        dataset = BdSLDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    model = Unet1D(dim=64, channels=args.input_dim * args.num_landmarks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Diffusion params
    timesteps = 1000
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x_start, labels) in enumerate(loader):
            # x_start: (batch, seq, landmarks, 3)
            batch_size = x_start.shape[0]
            
            # Rearrange to (Batch, Channels, Seq) for 1D Conv
            x_start = x_start.view(batch_size, 100, -1).permute(0, 2, 1).to(device)
            labels = labels.to(device)
            
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            noise = torch.randn_like(x_start)
            
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None]
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None]
            
            x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Predict noise
            noise_pred = model(x_t, t, labels)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")
        
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/diffusion_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/npy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=75)
    parser.add_argument("--use_mock", action="store_true", default=True)
    args = parser.parse_args()
    train(args)
