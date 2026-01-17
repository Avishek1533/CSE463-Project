import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SignTransformer(nn.Module):
    def __init__(self, num_classes=401, num_landmarks=75, input_dim=3, d_model=256, nhead=4, num_layers=4):
        super(SignTransformer, self).__init__()
        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        self.output_dim = num_landmarks * input_dim
        
        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Decoder
        # In a generative setting from class label, we can treat the class embedding as the "memory"
        # or initial token.
        # But commonly for sequence generation we use a standard decoder.
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc_out = nn.Linear(d_model, self.output_dim)
        
    def forward(self, labels, tgt_seq_len=100):
        # labels: (batch_size,)
        # Create a learnable query or simple expansion for the decoder
        batch_size = labels.size(0)
        
        # (Seq_len, Batch, D_model)
        # We can project the class label to the sequence length or just use it as memory
        # Here: we initialize the sequence with the class embedding (repeated) to condition it?
        # Or better: standard autoregressive is hard without ground truth inputs.
        # Let's do a non-autoregressive parallel generation for simplicity first, 
        # or simple expansion.
        
        label_embed = self.label_embedding(labels) # (Batch, D_model)
        
        # Expand to sequence length
        tgt = label_embed.unsqueeze(0).repeat(tgt_seq_len, 1, 1) # (Seq_len, Batch, D_model)
        
        # Add PE
        tgt = self.pos_encoder(tgt)
        
        # Pass through Decoder (Self-attention will relate frames)
        # memory is None because we are generating from the "tgt" seed directly
        output = self.transformer_decoder(tgt, memory=torch.zeros_like(tgt)) 
        
        output = self.fc_out(output) # (Seq_len, Batch, Out_dim)
        
        # Reshape to (Batch, Seq_len, Num_landmarks, 3)
        output = output.transpose(0, 1) # (Batch, Seq, Out_dim)
        output = output.view(batch_size, tgt_seq_len, self.num_landmarks, self.input_dim)
        
        return output
