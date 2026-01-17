import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=401, latent_dim=100, seq_len=100, num_landmarks=75, input_dim=3):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        self.output_flat = num_landmarks * input_dim
        
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # Generator structure
        # Input: Latent (100) + Label (50) = 150
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + 50, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # Output size: seq_len * output_flat (flattened sequence)
            nn.Linear(1024, seq_len * self.output_flat) # Simple MLP generation
        )
        
        # Alternatively could use ConvTranspose1d for temporal upsampling to be smarter
        
    def forward(self, noise, labels):
        # noise: (batch, latent_dim)
        # labels: (batch)
        label_input = self.label_emb(labels)
        gen_input = torch.cat((label_input, noise), -1)
        
        img = self.l1(gen_input)
        img = img.view(img.size(0), self.seq_len, self.num_landmarks, self.input_dim)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=401, seq_len=100, num_landmarks=75, input_dim=3):
        super(Discriminator, self).__init__()
        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        flat_dim = num_landmarks * input_dim
        
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # Flatten sequence for simple MLP discriminator or use Conv1D
        # Using MLP for simplicity first as baseline
        self.model = nn.Sequential(
            nn.Linear(seq_len * flat_dim + 50, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        # img: (batch, seq_len, num_landmarks, 3)
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)
        
        label_input = self.label_emb(labels)
        
        d_in = torch.cat((img_flat, label_input), -1)
        validity = self.model(d_in)
        return validity
