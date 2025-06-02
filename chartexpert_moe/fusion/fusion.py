import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartFusionModule(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Dynamic gating network
        # Input: concatenated features (visual_feat, text_feat, spatial_feat, semantic_feat) -> dim * 4
        # Output: 4 weights for each of the features
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 4),
            nn.Softmax(dim=-1)
        )
        
        # FILM conditioning for guided fusion
        # Generates gamma and beta from text_feat (dim) to modulate visual_feat (dim)
        # Output is dim * 2 (for gamma and beta, each of size dim)
        self.film_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU()
        )
        
        # Graph-based attention for structural relationships
        # Query: spatial_feat, Key: visual_modulated, Value: visual_modulated
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Assuming (batch, seq, feature) format
        )
        print(f"ChartFusionModule initialized: dim={dim}, num_heads={num_heads}, dropout={dropout}")

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor, 
                spatial_feat: torch.Tensor, semantic_feat: torch.Tensor) -> torch.Tensor:
        # Assuming all input features are of shape (batch_size, sequence_length, dim) or (batch_size, dim)
        # For MultiheadAttention, input needs to be (N, L, E) or (L, N, E) depending on batch_first.
        # Let's assume (N, L, E) where N is batch_size, L is sequence_length, E is embedding_dim (dim).
        # If features are global (N, E), we might need to unsqueeze them to (N, 1, E).
        
        # Validate shapes - assuming (N, L, E) for all for now.
        # If any feature is global (N,E), it should be unsqueezed to (N,1,E) before this module or handled explicitly.
        common_shape = visual_feat.shape
        for feat_name, feat_tensor in [('text', text_feat), ('spatial', spatial_feat), ('semantic', semantic_feat)]:
            if feat_tensor.shape != common_shape:
                raise ValueError(f"{feat_name}_feat shape {feat_tensor.shape} does not match visual_feat shape {common_shape}. All inputs must have the same shape (batch, seq_len, dim). Global features should be unsqueezed.")
            if feat_tensor.shape[-1] != self.dim:
                 raise ValueError(f"{feat_name}_feat dimension {feat_tensor.shape[-1]} does not match module's dim {self.dim}.")

        # Compute adaptive fusion weights
        # If inputs are (N, L, E), gate expects (N, L, E*4). We need to cat along the last dim.
        combined_for_gate = torch.cat([visual_feat, text_feat, spatial_feat, semantic_feat], dim=-1)
        weights = self.gate(combined_for_gate) # Shape: (N, L, 4)
        
        # Apply FILM modulation
        # film_params from text_feat (N, L, D) -> (N, L, D*2)
        film_params = self.film_generator(text_feat) # Shape: (N, L, dim*2)
        gamma, beta = film_params.chunk(2, dim=-1) # Each: (N, L, dim)
        
        visual_modulated = gamma * visual_feat + beta # Shape: (N, L, dim)
        
        # Graph-based structural fusion
        # self.graph_attention(query, key, value)
        # Query: spatial_feat (N, L, D)
        # Key: visual_modulated (N, L, D)
        # Value: visual_modulated (N, L, D)
        # Output: (output, attn_weights). output shape (N, L, D)
        structural_output, _ = self.graph_attention(
            query=spatial_feat, key=visual_modulated, value=visual_modulated
        )
        
        # Weighted combination
        # weights shape: (N, L, 4). Unsqueeze last dim to allow broadcast with features (N, L, D)
        # weights[..., 0:1] -> (N, L, 1)
        fused = (
            weights[..., 0:1] * visual_modulated + 
            weights[..., 1:2] * text_feat + 
            weights[..., 2:3] * structural_output + 
            weights[..., 3:4] * semantic_feat
        ) # Shape: (N, L, D)
        
        return fused
