import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import euclidean

def pad_sequence(sequence, max_len):
    """Pads a sequence of landmarks to max_len."""
    # sequence: (seq_len, num_landmarks, 3)
    seq_len = sequence.shape[0]
    if seq_len >= max_len:
        return sequence[:max_len]
    
    padding = np.zeros((max_len - seq_len, sequence.shape[1], sequence.shape[2]))
    return np.concatenate([sequence, padding], axis=0)

def calculate_mpje(predicted, target):
    """
    Mean Per Joint Position Error.
    predicted: (batch_size, seq_len, num_landmarks, 3) or (seq_len, num_landmarks, 3)
    target: same shape
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    diff = predicted - target
    dist = np.sqrt(np.sum(diff**2, axis=-1)) # (batch, seq, landmarks)
    return np.mean(dist)

def calculate_fid_proxy(real_features, fake_features):
    """
    Simple proxy for FID using mean and covariance of features.
    In a real scenario, use a pre-trained feature extractor (like InceptionV3 for images, 
    or a motion encoder for skeletons). Here we assume features are passed.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = (sigma1.dot(sigma2))**0.5
    
    # Check for imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def render_landmarks(landmarks, height=256, width=256):
    """
    Renders landmarks to a binary image (frame).
    landmarks: (num_landmarks, 3) -> we ignore Z for 2D rendering or project it.
    This is a simplification for SSIM calculation.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    # Simple projection: x, y are normalized [0, 1] usually in MediaPipe.
    # If not, assume they fit in frame.
    
    # Assuming standard MediaPipe structure (x, y, z)
    # We'll just plot points for now.
    for point in landmarks:
        x, y = int(point[0] * width), int(point[1] * height)
        if 0 <= x < width and 0 <= y < height:
            canvas[y, x] = 255
            
            # Optional: Draw small circle/cross for better visibility
            # ...
            
    return canvas

def calculate_ssim(predicted_seq, target_seq):
    """
    Calculates SSIM between rendered frames of predicted and real sequences.
    predicted_seq: (seq_len, num_landmarks, 3)
    target_seq: (seq_len, num_landmarks, 3)
    """
    if isinstance(predicted_seq, torch.Tensor):
        predicted_seq = predicted_seq.detach().cpu().numpy()
    if isinstance(target_seq, torch.Tensor):
        target_seq = target_seq.detach().cpu().numpy()
        
    ssim_scores = []
    # Limit frames to verify to save time if needed, or do all
    for i in range(min(len(predicted_seq), len(target_seq))):
        img_pred = render_landmarks(predicted_seq[i])
        img_targ = render_landmarks(target_seq[i])
        
        score, _ = ssim(img_pred, img_targ, full=True, data_range=255)
        ssim_scores.append(score)
        
    return np.mean(ssim_scores)
