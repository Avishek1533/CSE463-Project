import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import pad_sequence

class BdSLDataset(Dataset):
    def __init__(self, data_dir, max_len=100, transform=None):
        """
        Args:
            data_dir (str): Path to directory containing .npy files.
            max_len (int): Maximum sequence length to pad/truncate to.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_dir = data_dir
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
        self.max_len = max_len
        self.transform = transform
        
        # Check if files found
        if not self.file_paths:
            print(f"Warning: No .npy files found in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load .npy file
        # Expected shape: (frames, num_landmarks, 3) or similar
        data = np.load(file_path)
        
        # Basic Validation/Reshaping if needed
        # Assuming data is (frames, landmarks, 3)
        
        # Pad or Truncate
        data = pad_sequence(data, self.max_len)
        
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        
        # Flatten simple filename to use as a dummy class or ID if needed
        # For real usage, you'd parse metadata.csv
        label = 0 # Placeholder
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

class MockDataset(Dataset):
    """
    Mock dataset for local verification.
    Generates random sequences.
    """
    def __init__(self, num_samples=100, seq_len=100, num_landmarks=75, input_dim=3):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_landmarks = num_landmarks
        self.input_dim = input_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence
        data = np.random.randn(self.seq_len, self.num_landmarks, self.input_dim).astype(np.float32)
        data = torch.tensor(data)
        label = 0
        return data, label
