"""
ChartExpert-MoE: Main model implementation

This module implements the core ChartExpert-MoE model that combines vision encoder,
language model backbone, specialized expert modules, and dynamic routing.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .base_models import VisionEncoder, LLMBackbone
from .moe_layer import MoELayer
from experts import (
    LayoutDetectionExpert,
    OCRGroundingExpert,
    ScaleInterpretationExpert,
    GeometricPropertyExpert,
    TrendPatternExpert,
    QueryDeconstructionExpert,
    NumericalReasoningExpert,
    KnowledgeIntegrationExpert,
    VisualTextualAlignmentExpert,
    ChartToGraphExpert,
    ShallowReasoningExpert,
    DeepReasoningOrchestratorExpert
)
from routing import DynamicRouter
from fusion import MultiModalFusion


class ChartExpertMoE(nn.Module):
    """
    ChartExpert-MoE: A specialized Mixture-of-Experts Vision-Language Model
    for complex chart reasoning.
    
    The model consists of:
    1. Vision encoder for processing chart images
    2. Language model backbone for text understanding and generation
    3. Specialized expert modules for different aspects of chart reasoning
    4. Dynamic routing mechanism to select appropriate experts
    5. Advanced fusion strategies for multimodal integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Base models
        self.vision_encoder = VisionEncoder(config["vision_encoder"])
        self.llm_backbone = LLMBackbone(config["llm_backbone"])
        
        # Multimodal fusion
        self.fusion = MultiModalFusion(config["fusion"])
        
        # Expert modules - Visual-Spatial & Structural
        self.layout_expert = LayoutDetectionExpert(config["experts"]["layout"])
        self.ocr_expert = OCRGroundingExpert(config["experts"]["ocr"])
        self.scale_expert = ScaleInterpretationExpert(config["experts"]["scale"])
        self.geometric_expert = GeometricPropertyExpert(config["experts"]["geometric"])
        self.trend_expert = TrendPatternExpert(config["experts"]["trend"])
        
        # Expert modules - Semantic & Relational
        self.query_expert = QueryDeconstructionExpert(config["experts"]["query"])
        self.numerical_expert = NumericalReasoningExpert(config["experts"]["numerical"])
        self.integration_expert = KnowledgeIntegrationExpert(config["experts"]["integration"])
        
        # Expert modules - Cross-Modal Fusion & Reasoning
        self.alignment_expert = VisualTextualAlignmentExpert(config["experts"]["alignment"])
        self.chart_to_graph_expert = ChartToGraphExpert(config["experts"]["chart_to_graph"])
        
        # Expert modules - Cognitive Effort Modulation
        self.shallow_reasoning_expert = ShallowReasoningExpert(config["experts"]["shallow_reasoning"])
        self.orchestrator_expert = DeepReasoningOrchestratorExpert(config["experts"]["orchestrator"])
        
        # Collect all experts
        self.experts = [
            self.layout_expert,
            self.ocr_expert,
            self.scale_expert,
            self.geometric_expert,
            self.trend_expert,
            self.query_expert,
            self.numerical_expert,
            self.integration_expert,
            self.alignment_expert,
            self.chart_to_graph_expert,
            self.shallow_reasoning_expert,
            self.orchestrator_expert
        ]
        
        # MoE layer with dynamic routing
        self.moe_layer = MoELayer(
            experts=self.experts,
            router=DynamicRouter(config["routing"]),
            config=config["moe"]
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config["hidden_size"],
            config["vocab_size"]
        )
        
        # Projection from expert hidden size to LLM hidden size
        self.expert_to_llm_projection = nn.Linear(
            config["hidden_size"],
            config["llm_backbone"]["hidden_size"]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize projection layers
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ChartExpert-MoE
        
        Args:
            image: Chart image tensor [batch_size, channels, height, width]
            input_ids: Text input token ids [batch_size, seq_len]
            attention_mask: Attention mask for text [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs and auxiliary losses
        """
        batch_size = image.size(0)
        
        # Encode visual features
        visual_features = self.vision_encoder(image)  # [batch_size, num_patches, hidden_size]
        
        # Encode text features
        text_features = self.llm_backbone.encode(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # [batch_size, seq_len, hidden_size]
        
        # Initial multimodal fusion
        fused_features = self.fusion(
            visual_features=visual_features,
            text_features=text_features,
            attention_mask=attention_mask
        )  # [batch_size, total_seq_len, hidden_size]
        
        # MoE processing with expert routing
        moe_output = self.moe_layer(
            hidden_states=fused_features,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        expert_outputs = moe_output["expert_outputs"]  # [batch_size, total_seq_len, hidden_size]
        routing_weights = moe_output["routing_weights"]  # [batch_size, total_seq_len, num_experts]
        aux_loss = moe_output["aux_loss"]  # Scalar
        
        # Project expert outputs to LLM hidden size
        expert_outputs_llm = self.expert_to_llm_projection(expert_outputs)
        
        # Generate final output through LLM backbone
        logits = self.llm_backbone.generate_logits(
            hidden_states=expert_outputs_llm,
            attention_mask=attention_mask
        )  # [batch_size, seq_len, vocab_size]
        
        outputs = {
            "logits": logits,
            "routing_weights": routing_weights,
            "aux_loss": aux_loss,
            "visual_features": visual_features,
            "text_features": text_features,
            "fused_features": fused_features,
            "expert_outputs": expert_outputs
        }
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary MoE loss
            total_loss = loss + self.config.get("aux_loss_weight", 0.01) * aux_loss
            outputs["loss"] = total_loss
            outputs["lm_loss"] = loss
        
        return outputs
    
    def predict(
        self,
        image_path: str,
        query: str,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prediction for a chart image and query
        
        Args:
            image_path: Path to chart image
            query: Natural language query about the chart
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing prediction and analysis
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # Tokenize query
        inputs = self.llm_backbone.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Generate response
        with torch.no_grad():
            outputs = self.forward(
                image=image_tensor,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Decode response
            generated_ids = self.llm_backbone.generate(
                hidden_states=outputs["expert_outputs"],
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
            
            response = self.llm_backbone.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        
        return {
            "response": response,
            "routing_weights": outputs["routing_weights"],
            "expert_activations": self._analyze_expert_activations(outputs["routing_weights"]),
            "confidence": self._calculate_confidence(outputs["logits"])
        }
    
    def _analyze_expert_activations(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """Analyze which experts were most active"""
        expert_names = [
            "layout", "ocr", "scale", "geometric", "trend",
            "query", "numerical", "integration", "alignment", 
            "chart_to_graph", "shallow_reasoning", "orchestrator"
        ]
        
        # Average activation across sequence
        avg_activations = routing_weights.mean(dim=(0, 1))  # [num_experts]
        
        return {
            expert_names[i]: float(avg_activations[i])
            for i in range(len(expert_names))
        }
    
    def _calculate_confidence(self, logits: torch.Tensor) -> float:
        """Calculate prediction confidence"""
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        return float(torch.mean(max_probs))
    
    @classmethod
    def from_pretrained(cls, model_path: str, config_path: Optional[str] = None):
        """Load pretrained model"""
        import yaml
        
        if config_path is None:
            config_path = f"{model_path}/config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = cls(config)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model to disk"""
        import os
        import yaml
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), f"{save_path}/pytorch_model.bin")
        
        # Save configuration
        with open(f"{save_path}/config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Model saved to {save_path}") 