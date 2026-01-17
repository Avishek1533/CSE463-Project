import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BdSLDataset, MockDataset
from models.gan import Generator, Discriminator
from utils import calculate_mpje
import os
import argparse
import numpy as np

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    if args.use_mock:
        dataset = MockDataset(num_samples=100)
    else:
        dataset = BdSLDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Models
    generator = Generator(num_classes=401, seq_len=100).to(device)
    discriminator = Discriminator(num_classes=401, seq_len=100).to(device)
    
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        g_loss_total = 0
        d_loss_total = 0
        
        for i, (imgs, labels) in enumerate(loader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_g.zero_grad()
            
            z = torch.randn(batch_size, 100, device=device)
            gen_imgs = generator(z, labels)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs, labels), valid)
            
            g_loss.backward()
            opt_g.step()
            g_loss_total += g_loss.item()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            
            real_loss = criterion(discriminator(real_imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            opt_d.step()
            d_loss_total += d_loss.item()
            
        print(f"[Epoch {epoch+1}/{args.epochs}] [D loss: {d_loss_total/len(loader):.4f}] [G loss: {g_loss_total/len(loader):.4f}]")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(min(16, args.batch_size), 100, device=device)
                sample_labels = torch.randint(0, 401, (min(16, args.batch_size),), device=device)
                gen_imgs = generator(z, sample_labels)
                # Ensure shapes match for MPJE, though MPJE comparison requires ground truth.
                # In GAN, we check diversity or visual quality. 
                # Calculating MPJE against random real samples of SAME class would be good.
                # For now just print "generated".
            
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(generator.state_dict(), f"checkpoints/gan_generator_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/npy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--use_mock", action="store_true", default=True)
    args = parser.parse_args()
    train(args)
