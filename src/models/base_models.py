"""
Base model components for ChartExpert-MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    CLIPVisionModel, SiglipVisionModel
)
import timm
import math


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for better spatial understanding in charts
    """
    def __init__(self, dim: int, max_resolution: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_resolution = max_resolution
        
        # Compute inverse frequencies
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            height: Height of the 2D grid
            width: Width of the 2D grid
        """
        batch_size, seq_len, dim = x.shape
        
        # Create 2D position indices
        y_pos = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)
        
        # Flatten positions
        y_pos = y_pos.reshape(-1)  # [height * width]
        x_pos = x_pos.reshape(-1)  # [height * width]
        
        # Apply RoPE separately for x and y coordinates
        sincos_y = torch.einsum('i,j->ij', y_pos.float(), self.inv_freq)
        sincos_x = torch.einsum('i,j->ij', x_pos.float(), self.inv_freq)
        
        sin_y, cos_y = sincos_y.sin(), sincos_y.cos()
        sin_x, cos_x = sincos_x.sin(), sincos_x.cos()
        
        # Combine x and y embeddings
        sin_pos = torch.cat([sin_y, sin_x], dim=-1)  # [seq_len, dim/2]
        cos_pos = torch.cat([cos_y, cos_x], dim=-1)  # [seq_len, dim/2]
        
        # Apply rotation
        x_reshaped = x.reshape(batch_size, seq_len, -1, 2)
        x_rotated = torch.stack([
            x_reshaped[..., 0] * cos_pos[:seq_len, :x_reshaped.size(-2)] - 
            x_reshaped[..., 1] * sin_pos[:seq_len, :x_reshaped.size(-2)],
            x_reshaped[..., 0] * sin_pos[:seq_len, :x_reshaped.size(-2)] + 
            x_reshaped[..., 1] * cos_pos[:seq_len, :x_reshaped.size(-2)]
        ], dim=-1)
        
        return x_rotated.reshape(batch_size, seq_len, dim)


class VisionEncoder(nn.Module):
    """
    Vision encoder module for processing chart images
    
    Supports multiple vision backbones:
    - CLIP/SigLIP
    - DINOv2  
    - MoonViT (with native resolution and 2D RoPE)
    - SAM (for edge detection)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.encoder_type = config["encoder_type"]
        self.hidden_size = config["hidden_size"]
        self.use_native_resolution = config.get("use_native_resolution", False)
        self.use_2d_rope = config.get("use_2d_rope", False)
        
        # Initialize encoder based on type
        if self.encoder_type == "clip":
            try:
                from transformers import CLIPVisionModel
                # Load without device map to avoid version issues
                self.encoder = CLIPVisionModel.from_pretrained(
                    config["model_name"],
                    torch_dtype=torch.float32
                )
                self.encoder_hidden_size = self.encoder.config.hidden_size
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                print("Using mock encoder for testing")
                self.encoder = self._create_mock_encoder()
                self.encoder_hidden_size = 768
        elif self.encoder_type == "dinov2":
            # DINOv2 for robust feature extraction
            try:
                self.encoder = self._load_dinov2(config["model_name"])
                self.encoder_hidden_size = 768  # DINOv2 base
            except Exception as e:
                print(f"Warning: Failed to load DINOv2: {e}")
                self.encoder = self._create_mock_encoder()
                self.encoder_hidden_size = 768
        elif self.encoder_type == "moonvit":
            # MoonViT with native resolution support
            self.encoder = self._load_moonvit(config)
            self.encoder_hidden_size = config.get("moonvit_hidden_size", 768)
        elif self.encoder_type == "sam":
            # SAM for edge-aware features
            try:
                self.encoder = self._load_sam(config["model_name"])
                self.encoder_hidden_size = 256  # SAM encoder
            except Exception as e:
                print(f"Warning: Failed to load SAM: {e}")
                self.encoder = self._create_mock_encoder()
                self.encoder_hidden_size = 768
        else:
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(config["model_name"])
                self.encoder_hidden_size = self.encoder.config.hidden_size
            except Exception as e:
                print(f"Warning: Failed to load model: {e}")
                self.encoder = self._create_mock_encoder()
                self.encoder_hidden_size = 768
        
        # Projection layer
        self.projection = nn.Linear(self.encoder_hidden_size, self.hidden_size)
        
        # 2D RoPE for better spatial understanding
        if self.use_2d_rope:
            self.rope_2d = RotaryPositionEmbedding2D(
                dim=self.hidden_size,
                max_resolution=config.get("max_resolution", 1024)
            )
    
    def _load_moonvit(self, config: Dict[str, Any]) -> nn.Module:
        """Load MoonViT with native resolution processing"""
        # Simplified MoonViT implementation
        class MoonViTEncoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.patch_size = config.get("patch_size", 16)
                self.hidden_size = config.get("moonvit_hidden_size", 768)
                
                # Variable resolution patch embedding
                self.patch_embed = nn.Conv2d(
                    3, self.hidden_size, 
                    kernel_size=self.patch_size, 
                    stride=self.patch_size
                )
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=12,
                        dim_feedforward=3072,
                        batch_first=True
                    ) for _ in range(12)
                ])
                
                self.norm = nn.LayerNorm(self.hidden_size)
            
            def forward(self, x):
                # x: [B, C, H, W]
                x = self.patch_embed(x)  # [B, D, H', W']
                B, D, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # [B, H'*W', D]
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                return x, (H, W)
        
        return MoonViTEncoder(config)
    
    def _load_dinov2(self, model_name: str) -> nn.Module:
        """Load DINOv2 model"""
        # Use timm to load DINOv2
        model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        return model
    
    def _load_sam(self, model_name: str) -> nn.Module:
        """Load SAM encoder"""
        # Mock SAM encoder for now
        class SAMEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 256, kernel_size=16, stride=16)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=256,
                        nhead=8,
                        dim_feedforward=1024,
                        batch_first=True
                    ) for _ in range(6)
                ])
                self.norm = nn.LayerNorm(256)
                
            def forward(self, x):
                x = self.patch_embed(x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                
                for block in self.blocks:
                    x = block(x)
                    
                x = self.norm(x)
                return x
                
        return SAMEncoder()
    
    def _get_encoder_hidden_size(self) -> int:
        """Get the hidden size of the encoder"""
        if hasattr(self.encoder, 'config'):
            return self.encoder.config.hidden_size
        elif hasattr(self.encoder, 'embed_dim'):
            return self.encoder.embed_dim
        elif hasattr(self.encoder, 'num_features'):
            return self.encoder.num_features
        else:
            return 768  # Default
    
    def _build_position_embedding(self, config: Dict[str, Any]) -> nn.Module:
        """Build 2D position embedding for spatial information"""
        max_patches = config.get("max_patches", 196)  # 14x14 for patch16 on 224x224
        return nn.Embedding(max_patches, self.hidden_size)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode chart images
        
        Args:
            images: Image tensor [batch_size, channels, height, width]
            
        Returns:
            Visual features [batch_size, num_patches, hidden_size]
        """
        batch_size = images.size(0)
        
        if self.encoder_type in ["clip", "siglip"]:
            # For CLIP/SigLIP, check if it's a real model or mock
            if hasattr(self.encoder, 'forward_features'):
                visual_features = self.encoder.forward_features(images)
            elif hasattr(self.encoder, 'vision_model'):
                # Real CLIP model
                outputs = self.encoder(pixel_values=images)
                if hasattr(outputs, 'last_hidden_state'):
                    visual_features = outputs.last_hidden_state
                else:
                    visual_features = outputs.pooler_output.unsqueeze(1)
            else:
                # Mock encoder
                visual_features = self.encoder(images)
        else:
            # For other encoders (DINOv2, MoonViT, SAM, mock)
            if hasattr(self.encoder, 'forward_features'):
                visual_features = self.encoder.forward_features(images)
            else:
                # Direct forward for mock or custom encoders
                visual_features = self.encoder(images)
                
        if len(visual_features.shape) == 2:
            # Add sequence dimension if needed
            visual_features = visual_features.unsqueeze(1)
        
        # Handle MoonViT output
        if isinstance(visual_features, tuple):
            visual_features = visual_features[0]
        
        # Project to target hidden size
        visual_features = self.projection(visual_features)
        
        # Add position embeddings if needed
        if hasattr(self, 'position_embedding'):
            seq_len = visual_features.size(1)
            position_ids = torch.arange(seq_len, device=visual_features.device)
            position_embeddings = self.position_embedding(position_ids)
            visual_features = visual_features + position_embeddings.unsqueeze(0)
        
        # Apply 2D RoPE if enabled
        if self.use_2d_rope and visual_features.size(1) > 1:
            # Assume square patches
            num_patches = visual_features.size(1)
            height = width = int(math.sqrt(num_patches))
            visual_features = self.rope_2d(visual_features, height, width)
        
        return visual_features

    def _align_features(self, features: torch.Tensor, target_len: int) -> torch.Tensor:
        """Align features to target sequence length"""
        current_len = features.size(1)
        
        if current_len == target_len:
            return features
        elif current_len > target_len:
            return features[:, :target_len, :]
        else:
            # Pad with zeros
            padding = torch.zeros(
                features.size(0),
                target_len - current_len,
                features.size(2),
                device=features.device
            )
            return torch.cat([features, padding], dim=1)
    
    def _create_mock_encoder(self) -> nn.Module:
        """Create a mock encoder for testing"""
        class MockEncoder(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.conv1 = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
                self.ln = nn.LayerNorm(hidden_size)
                
            def forward(self, x):
                # x: [B, C, H, W]
                x = self.conv1(x)  # [B, hidden_size, H', W']
                x = x.flatten(2).transpose(1, 2)  # [B, H'*W', hidden_size]
                x = self.ln(x)
                return x
                
        return MockEncoder(768)  # Use default 768


class LLMBackbone(nn.Module):
    """
    Large Language Model backbone supporting multiple architectures
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get("model_name", "meta-llama/Llama-2-7b-hf")
        self.hidden_size = config.get("hidden_size", 4096)
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model for encoding
            self.encoder = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if config.get("use_fp16", False) else torch.float32,
                device_map="auto" if config.get("use_device_map", False) else None
            )
            
            # Load causal LM for generation
            self.generator = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if config.get("use_fp16", False) else torch.float32,
                device_map="auto" if config.get("use_device_map", False) else None
            )
        except Exception as e:
            print(f"Warning: Failed to load LLM model: {e}")
            print("Using mock LLM for testing")
            self._create_mock_llm()
        
        # Adaptation layers for multimodal input
        self.multimodal_projection = nn.Linear(
            config.get("vision_hidden_size", 768),
            self.hidden_size
        )
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text input
        
        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text features [batch_size, seq_len, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.last_hidden_state
    
    def generate_logits(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate logits from hidden states
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Use the generator's language modeling head
        logits = self.generator.lm_head(hidden_states)
        return logits
    
    def generate(
        self,
        hidden_states: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from hidden states
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated token ids [batch_size, generated_seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Convert hidden states to embeddings that can be used for generation
        # This is a simplified approach; in practice, you might need more sophisticated handling
        input_embeds = hidden_states
        
        # Generate using the causal LM
        outputs = self.generator.generate(
            inputs_embeds=input_embeds,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return outputs
    
    def prepare_multimodal_input(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare multimodal input by concatenating text and visual features
        
        Args:
            text_features: Text features [batch_size, text_seq_len, hidden_size]
            visual_features: Visual features [batch_size, visual_seq_len, vision_hidden_size]
            attention_mask: Text attention mask [batch_size, text_seq_len]
            
        Returns:
            Tuple of (combined_features, combined_attention_mask)
        """
        batch_size = text_features.size(0)
        text_seq_len = text_features.size(1)
        visual_seq_len = visual_features.size(1)
        
        # Project visual features to text hidden size
        projected_visual = self.multimodal_projection(visual_features)
        
        # Concatenate features
        combined_features = torch.cat([projected_visual, text_features], dim=1)
        
        # Create combined attention mask
        visual_attention = torch.ones(
            batch_size, visual_seq_len,
            device=visual_features.device,
            dtype=torch.long
        )
        
        if attention_mask is not None:
            combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
        else:
            text_attention = torch.ones(
                batch_size, text_seq_len,
                device=text_features.device,
                dtype=torch.long
            )
            combined_attention_mask = torch.cat([visual_attention, text_attention], dim=1)
        
        return combined_features, combined_attention_mask
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings if needed"""
        self.encoder.resize_token_embeddings(new_num_tokens)
        self.generator.resize_token_embeddings(new_num_tokens)
    
    def _create_mock_llm(self):
        """Create mock LLM components for testing"""
        # Mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = 32000
                
            def __len__(self):
                return self.vocab_size
                
            def __call__(self, text, **kwargs):
                # Mock tokenization
                return {
                    "input_ids": torch.randint(0, self.vocab_size, (1, 50)),
                    "attention_mask": torch.ones(1, 50)
                }
                
            def decode(self, token_ids, **kwargs):
                return "Mock decoded text"
        
        # Mock encoder
        class MockEncoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.embed = nn.Embedding(32000, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=16,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True
                    ),
                    num_layers=4
                )
                
            def forward(self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
                x = self.embed(input_ids)
                x = self.transformer(x)
                
                class Output:
                    def __init__(self, last_hidden_state):
                        self.last_hidden_state = last_hidden_state
                        
                return Output(x)
        
        # Mock generator
        class MockGenerator(nn.Module):
            def __init__(self, hidden_size, vocab_size):
                super().__init__()
                self.hidden_size = hidden_size  # Store hidden_size
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def generate(self, inputs_embeds, **kwargs):
                # Mock generation
                batch_size = inputs_embeds.size(0)
                max_length = kwargs.get("max_length", 100)
                return torch.randint(0, 32000, (batch_size, max_length))
        
        self.tokenizer = MockTokenizer()
        self.encoder = MockEncoder(self.hidden_size)
        self.generator = MockGenerator(self.hidden_size, 32000) 