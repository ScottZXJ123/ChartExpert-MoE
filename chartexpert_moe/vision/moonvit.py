import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Placeholder for loading pretrained model
def load_pretrained(base_model_name: str) -> nn.Module:
    print(f"Placeholder: load_pretrained('{base_model_name}') - returning a dummy nn.Module. Replace with actual model loading and ensure output shape is (B, C_in, H_feat, W_feat) for PixelShuffle.")
    
    # This dummy model will output (B, C, H, W) for PixelShuffle to work.
    # C_in for PixelShuffle must be hidden_dim * (downscale_factor^2)
    # If hidden_dim = 768 and downscale_factor = 2, C_in = 768 * 4 = 3072.
    # H_feat, W_feat are placeholder feature map dimensions.
    class DummyVisionEncoder(nn.Module):
        def __init__(self, out_channels, h, w):
            super().__init__()
            self.out_channels = out_channels
            self.h = h
            self.w = w
            # A dummy conv layer to simulate some processing and parameter registration
            self.conv = nn.Conv2d(3, out_channels, kernel_size=1, padding=0) 
            print(f"DummyVisionEncoder initialized to output: (B, {out_channels}, {h}, {w})")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is assumed to be (B, 3, H_img, W_img).
            b = x.shape[0]
            # Create a dummy tensor of the correct input size for the conv layer
            # if we want the conv output to be B, self.out_channels, self.h, self.w
            # and conv is K=1, S=1, P=0
            dummy_conv_input = torch.randn(b, 3, self.h, self.w, device=x.device)
            return self.conv(dummy_conv_input)

    # hidden_dim (output of PixelShuffle) = 768 (as in MoonViTEncoder default)
    # downscale_factor = 2
    # C_in_pixelshuffle = hidden_dim * (downscale_factor**2) = 768 * 4 = 3072
    # H_feat, W_feat for dummy encoder output, e.g., 16, 16 (these are H,W *before* pixel shuffle's upscale of spatial dim)
    return DummyVisionEncoder(out_channels=3072, h=16, w=16)

class RoPE2D(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base_theta: float = 10000.0, learnable_freq: bool = False):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base_theta = base_theta
        
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")

        inv_freq = 1.0 / (base_theta ** (torch.arange(0, dim, 2).float() / dim))
        if learnable_freq:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        seq_len = x.shape[1]
        
        if seq_len > self.max_seq_len:
            # This case should be handled by padding/truncation in MoonViTEncoder's forward
            raise ValueError(f"Sequence length {seq_len} exceeds RoPE maximum sequence length {self.max_seq_len}")
        
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input feature dimension {x.shape[-1]} does not match RoPE dimension {self.dim}")

        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]
        
        half_dim = self.dim // 2
        x_left_half = x[..., :half_dim]
        x_right_half = x[..., half_dim:]
        
        x_emb_rotated_halves = torch.cat([-x_right_half, x_left_half], dim=-1)
        return (x * cos) + (x_emb_rotated_halves * sin)


class MoonViTEncoder(nn.Module):
    def __init__(self, base_model='SigLIP-SO-400M', hidden_dim=768, llm_dim=4096, 
                 max_patches=256, rope_base_theta=10000.0, learnable_rope_freq=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.llm_dim = llm_dim
        self.max_patches = max_patches

        self.vision_encoder = load_pretrained(base_model)
        self.pixel_shuffle = nn.PixelShuffle(downscale_factor=2)
        
        self.rope_2d = RoPE2D(
            dim=self.hidden_dim,
            max_seq_len=self.max_patches,
            base_theta=rope_base_theta,
            learnable_freq=learnable_rope_freq
        )
        
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )
        print(f"MoonViTEncoder initialized: hidden_dim={hidden_dim}, llm_dim={llm_dim}, max_patches={max_patches}. RoPE learnable_freq={learnable_rope_freq}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: input image tensor (e.g., B, 3, H_img, W_img)
        
        features = self.vision_encoder(x) 
        # Expected output from dummy vision_encoder: (B, self.hidden_dim * 4, H_feat, W_feat)
        # e.g., (B, 3072, 16, 16)
        
        features_shuffled = self.pixel_shuffle(features)
        # Expected output: (B, self.hidden_dim, H_feat * 2, W_feat * 2)
        # e.g., (B, 768, 32, 32)
        
        batch_size, channels, height, width = features_shuffled.shape
        patch_embeddings = features_shuffled.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        # Expected shape: (B, num_patches, self.hidden_dim) e.g., (B, 32*32=1024, 768)
        
        current_num_patches = patch_embeddings.shape[1]
        if current_num_patches != self.max_patches:
            if current_num_patches > self.max_patches:
                patch_embeddings = patch_embeddings[:, :self.max_patches, :]
            else: # current_num_patches < self.max_patches
                padding_size = self.max_patches - current_num_patches
                padding = torch.zeros(batch_size, padding_size, channels, 
                                      device=patch_embeddings.device, dtype=patch_embeddings.dtype)
                patch_embeddings = torch.cat([patch_embeddings, padding], dim=1)
        # Now patch_embeddings is (B, self.max_patches, self.hidden_dim)
        
        patch_embeddings_rotated = self.rope_2d(patch_embeddings)
        projected_embeddings = self.projector(patch_embeddings_rotated)
        # Expected shape: (B, self.max_patches, self.llm_dim)
        
        return projected_embeddings
