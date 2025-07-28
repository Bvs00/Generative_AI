import torch.nn as nn
import torch
import os
import math
from cbam import CBAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to check if a file whose name contains base_filename exists in folder
def find_matching_checkpoint(checkpoint_folder, base_filename):
    for fname in os.listdir(checkpoint_folder):
        name, ext = os.path.splitext(fname)
        if base_filename in name:
            return os.path.join(checkpoint_folder, fname)
    return None

class NoiseSchedule:
    def __init__(self, L, s=0.008):
        self.L=L
        t=torch.linspace(0.0, L, L+1, device=device)/L
        a=torch.cos((t+s)/(1+s)*torch.pi/2)**2
        a=a/a[0]
        self.beta=(1-a[1:]/a[:-1]).clip(0.0, 0.99)
        self.alpha=torch.cumprod(1.0-self.beta, dim=0)
        self.one_minus_beta=1-self.beta
        self.one_minus_alpha=1-self.alpha
        self.sqrt_alpha=torch.sqrt(self.alpha)
        self.sqrt_beta=torch.sqrt(self.beta)
        self.sqrt_1_alpha=torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta=torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L


class TimeEncoding:
    def __init__(self, L, dim):
        # Note: the dimension dim should be an even number
        self.L=L
        self.dim=dim
        dim2=dim//2
        encoding=torch.zeros(L, dim)
        ang=torch.linspace(0.0, torch.pi/2, L)
        logmul=torch.linspace(0.0, math.log(40), dim2)
        mul=torch.exp(logmul)
        for i in range(dim2):
            a=ang*mul[i]
            encoding[:,2*i]=torch.sin(a)
            encoding[:,2*i+1]=torch.cos(a)
        self.encoding=encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(2).unsqueeze(3)
        beta = self.beta(cond).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)

# --- Self-Attention Block ---
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        x_norm = self.norm(x_flat)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out  # Residual



class DecoderWithFiLM(nn.Module):
    def __init__(self, from_features, to_features, cond_features):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(from_features, from_features, 5, padding='same', bias=False),
                nn.BatchNorm2d(from_features),
                nn.ReLU(),
                CBAM(from_features),  # Attention module  
            )
        self.film = FiLM(cond_features, from_features)
        self.conv2=nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            CBAM(from_features),  # Attention module                
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
            )

    def forward(self, x, cond):
        x = self.conv1(x)
        x = self.film(x, cond)
        x = self.conv2(x)
        return x

# --- Encoder Module ---
class EncoderWithFilm(nn.Module):
    def __init__(self, from_features, to_features, cond_features):
        super().__init__()
        
        self.conv1=nn.Sequential(
                nn.Conv2d(from_features, from_features, 5, padding='same', bias=False),
                nn.BatchNorm2d(from_features),
                nn.ReLU(),
                CBAM(from_features),  # Attention module
            )
        self.film = FiLM(cond_features, from_features)
        self.conv2=nn.Sequential(
                nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                nn.BatchNorm2d(from_features),
                nn.ReLU(),
                CBAM(from_features),  # Attention module
                nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(to_features),
                nn.ReLU()
            )

    def forward(self, x, cond):
        x = self.conv1(x)
        x = self.film(x, cond)
        x = self.conv2(x)
        return x