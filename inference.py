import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from models.transformer import SignTransformer
from models.gan import Generator
from models.diffusion import Unet1D, cosine_beta_schedule
from utils import render_landmarks

def generate_transformer(model, label, device):
    model.eval()
    label = torch.tensor([label]).to(device)
    with torch.no_grad():
        output = model(label, tgt_seq_len=100)
    return output.squeeze(0).cpu().numpy()

def generate_gan(model, label, device):
    model.eval()
    z = torch.randn(1, 100).to(device)
    label = torch.tensor([label]).to(device)
    with torch.no_grad():
        output = model(z, label)
    return output.squeeze(0).cpu().numpy()

def generate_diffusion(model, label, device, timesteps=1000):
    model.eval()
    label = torch.tensor([label]).to(device)
    
    # Sampling Loop
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Posterior variance
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    # Start from noise
    # Shape: (1, Channels, Seq)
    img = torch.randn(1, 75*3, 100).to(device)
    
    for i in reversed(range(0, timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(img, t, label)
            
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[i])
        sqrt_recip_alpha_t = sqrt_recip_alphas[i]
        
        mean = sqrt_recip_alpha_t * (img - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
        
        if i > 0:
            noise = torch.randn_like(img)
            var = torch.sqrt(posterior_variance[i]) * noise
        else:
            var = 0.
            
        img = mean + var
        
    img = img.permute(0, 2, 1).view(1, 100, 75, 3) # (B, Seq, Landmarks, 3)
    return img.squeeze(0).cpu().numpy()
    
import torch.nn.functional as F # Re-import for safety inside function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["transformer", "gan", "diffusion"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label", type=int, default=0, help="Class label to generate")
    parser.add_argument("--output_path", type=str, default="output.npy")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type == "transformer":
        model = SignTransformer(num_classes=401).to(device)
    elif args.model_type == "gan":
        model = Generator(num_classes=401, seq_len=100).to(device)
    elif args.model_type == "diffusion":
        model = Unet1D(dim=64, channels=75*3).to(device)
        
    # Load weights
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Checkpoint loaded.")
    else:
        print("Checkpoint not found, using random weights.")
        
    print(f"Generating for label {args.label} using {args.model_type}...")
    
    if args.model_type == "transformer":
        output = generate_transformer(model, args.label, device)
    elif args.model_type == "gan":
        output = generate_gan(model, args.label, device)
    elif args.model_type == "diffusion":
        output = generate_diffusion(model, args.label, device)
        
    np.save(args.output_path, output)
    print(f"Saved generated sequence to {args.output_path} with shape {output.shape}")
    
    # Verification of shape
    assert output.shape == (100, 75, 3), f"Output shape mismatch: {output.shape}"
