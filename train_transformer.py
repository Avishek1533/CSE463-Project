import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BdSLDataset, MockDataset
from models.transformer import SignTransformer
from utils import calculate_mpje
import os
import argparse

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset Setup
    if args.use_mock:
        dataset = MockDataset(num_samples=100)
    else:
        dataset = BdSLDataset(args.data_dir)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model Setup
    model = SignTransformer(num_classes=401).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(device) # (B, Seq, Landmarks, 3)
            labels = labels.to(device)
            
            # Forward
            # For Transformer, we generate sequence from labels
            # Output shape: (B, Seq, Landmarks, 3)
            output = model(labels, tgt_seq_len=data.shape[1])
            
            loss = criterion(output, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        # Validation / Metrics
        model.eval()
        with torch.no_grad():
            # Just take the last batch for quick metric check or run full validation set
            metrics_mpje = calculate_mpje(output, data)
            
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | MPJE: {metrics_mpje:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/transformer_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/npy", help="Path to .npy files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_mock", action="store_true", default=True, help="Use mock data for testing")
    args = parser.parse_args()
    
    train(args)
