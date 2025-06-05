"""
Dynamic Router for ChartExpert-MoE

Implements intelligent routing mechanisms that dynamically select appropriate experts
based on input content, modality, and task requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Tuple
import math


class DynamicRouter(nn.Module):
    """
    Dynamic router that selects experts based on input characteristics
    
    Features:
    - Content-aware routing based on hidden states
    - Modality-aware routing (visual vs textual)
    - Context-sensitive expert selection
    - Load balancing mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)
        self.num_experts = config.get("num_experts", 10)
        self.routing_strategy = config.get("routing_strategy", "learned")
        self.top_k = config.get("top_k", 2)
        
        # Expert type definitions
        self.expert_types = [
            "layout", "ocr", "scale", "geometric", "trend",
            "query", "numerical", "integration", "alignment", "orchestrator"
        ]
        
        # Learned routing components
        if self.routing_strategy == "learned":
            self.content_router = ContentAwareRouter(config)
            self.modality_router = ModalityAwareRouter(config)
            self.context_router = ContextSensitiveRouter(config)
            
            # Fusion of different routing signals
            self.routing_fusion = nn.Linear(self.num_experts * 3, self.num_experts)
        
        # Task-specific routing patterns (if using rule-based routing)
        elif self.routing_strategy == "rule_based":
            self.task_patterns = self._define_task_patterns()
        
        # Hybrid approach
        elif self.routing_strategy == "hybrid":
            self.content_router = ContentAwareRouter(config)
            self.task_patterns = self._define_task_patterns()
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # Learnable fusion weight
        
        # Load balancing
        self.load_balancing = config.get("load_balancing", True)
        if self.load_balancing:
            self.register_buffer('expert_usage', torch.zeros(self.num_experts))
            self.balance_weight = config.get("balance_weight", 0.01)
    
    def _define_task_patterns(self) -> Dict[str, List[int]]:
        """Define expert activation patterns for different task types"""
        return {
            "visual_comparison": [0, 3, 4],  # layout, geometric, trend
            "text_extraction": [1, 2, 5],   # ocr, scale, query  
            "numerical_reasoning": [2, 6, 9], # scale, numerical, orchestrator
            "spatial_analysis": [0, 3, 8],   # layout, geometric, alignment
            "trend_analysis": [4, 6, 9],     # trend, numerical, orchestrator
            "complex_reasoning": [5, 6, 7, 9] # query, numerical, integration, orchestrator
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Route inputs to appropriate experts
        
        Args:
            hidden_states: Input features [batch_size * seq_len, hidden_size]
            image: Optional image tensor
            input_ids: Optional input token ids
            attention_mask: Optional attention mask
            task_type: Optional explicit task type
            
        Returns:
            Dictionary containing routing logits and metadata
        """
        batch_size = hidden_states.size(0)
        
        if self.routing_strategy == "learned":
            routing_logits = self._learned_routing(
                hidden_states, image, input_ids, attention_mask
            )
        elif self.routing_strategy == "rule_based":
            routing_logits = self._rule_based_routing(
                hidden_states, image, input_ids, attention_mask, task_type
            )
        elif self.routing_strategy == "hybrid":
            routing_logits = self._hybrid_routing(
                hidden_states, image, input_ids, attention_mask, task_type
            )
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
        
        # Apply load balancing if enabled
        if self.load_balancing and self.training:
            routing_logits = self._apply_load_balancing(routing_logits)
        
        return {
            "logits": routing_logits,
            "routing_weights": F.softmax(routing_logits, dim=-1),
            "expert_types": self.expert_types
        }
    
    def _learned_routing(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Learned routing using neural networks"""
        # Content-based routing
        content_logits = self.content_router(hidden_states)
        
        # Modality-based routing
        modality_logits = self.modality_router(hidden_states, image, input_ids)
        
        # Context-based routing
        context_logits = self.context_router(hidden_states, attention_mask)
        
        # Fuse different routing signals
        combined_logits = torch.cat([content_logits, modality_logits, context_logits], dim=-1)
        routing_logits = self.routing_fusion(combined_logits)
        
        return routing_logits
    
    def _rule_based_routing(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        task_type: Optional[str]
    ) -> torch.Tensor:
        """Rule-based routing using predefined patterns"""
        batch_size = hidden_states.size(0)
        routing_logits = torch.zeros(batch_size, self.num_experts, device=hidden_states.device)
        
        # Determine task type if not provided
        if task_type is None:
            task_type = self._infer_task_type(hidden_states, image, input_ids)
        
        # Apply task-specific pattern
        if task_type in self.task_patterns:
            expert_indices = self.task_patterns[task_type]
            for idx in expert_indices:
                routing_logits[:, idx] = 1.0
        else:
            # Default to all experts with equal weight
            routing_logits.fill_(1.0)
        
        return routing_logits
    
    def _hybrid_routing(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        task_type: Optional[str]
    ) -> torch.Tensor:
        """Hybrid routing combining learned and rule-based approaches"""
        # Get learned routing
        learned_logits = self.content_router(hidden_states)
        
        # Get rule-based routing
        rule_logits = self._rule_based_routing(
            hidden_states, image, input_ids, attention_mask, task_type
        )
        
        # Combine using learnable weight
        fusion_weight = torch.sigmoid(self.fusion_weight)
        routing_logits = fusion_weight * learned_logits + (1 - fusion_weight) * rule_logits
        
        return routing_logits
    
    def _infer_task_type(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor]
    ) -> str:
        """Infer task type from input characteristics"""
        # Simple heuristics for task type inference
        # In practice, this could be more sophisticated
        
        if image is not None and input_ids is not None:
            # Multimodal input - likely complex reasoning
            return "complex_reasoning"
        elif image is not None:
            # Visual input dominant - likely visual analysis
            return "visual_comparison"
        else:
            # Text input dominant - likely numerical reasoning
            return "numerical_reasoning"
    
    def _apply_load_balancing(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """Apply load balancing to encourage expert diversity"""
        if not hasattr(self, 'expert_usage'):
            return routing_logits
        
        # Calculate usage imbalance penalty
        total_usage = self.expert_usage.sum()
        if total_usage > 0:
            usage_probs = self.expert_usage / total_usage
            usage_penalty = usage_probs.to(routing_logits.device)
            
            # Reduce logits for overused experts
            routing_logits = routing_logits - self.balance_weight * usage_penalty.unsqueeze(0)
        
        return routing_logits
    
    def update_expert_usage(self, expert_selections: torch.Tensor):
        """Update expert usage statistics"""
        if self.load_balancing:
            usage_counts = torch.bincount(expert_selections.flatten(), minlength=self.num_experts)
            self.expert_usage += usage_counts.cpu()


class ContentAwareRouter(nn.Module):
    """Router that makes decisions based on input content"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_experts = config.get("num_experts", 10)
        
        # Content analysis layers
        self.content_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_experts)
        )
        
        # Attention mechanism for content focus
        self.content_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Content-aware routing"""
        # Apply self-attention to focus on important content
        hidden_states_seq = hidden_states.unsqueeze(1)
        attended_content, _ = self.content_attention(
            hidden_states_seq, hidden_states_seq, hidden_states_seq
        )
        attended_content = attended_content.squeeze(1)
        
        # Analyze content for expert selection
        routing_logits = self.content_analyzer(attended_content)
        
        return routing_logits


class ModalityAwareRouter(nn.Module):
    """Router that considers input modality (visual vs textual)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_experts = config.get("num_experts", 10)
        
        # Modality detection
        self.modality_detector = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # visual, textual, multimodal
        )
        
        # Modality-specific routing
        self.visual_router = nn.Linear(self.hidden_size, self.num_experts)
        self.textual_router = nn.Linear(self.hidden_size, self.num_experts)
        self.multimodal_router = nn.Linear(self.hidden_size, self.num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Modality-aware routing"""
        # Detect modality
        modality_logits = self.modality_detector(hidden_states)
        modality_probs = F.softmax(modality_logits, dim=-1)
        
        # Get routing from each modality-specific router
        visual_routing = self.visual_router(hidden_states)
        textual_routing = self.textual_router(hidden_states)
        multimodal_routing = self.multimodal_router(hidden_states)
        
        # Weight by modality probabilities
        routing_logits = (
            modality_probs[:, 0:1] * visual_routing +
            modality_probs[:, 1:2] * textual_routing +
            modality_probs[:, 2:3] * multimodal_routing
        )
        
        return routing_logits


class ContextSensitiveRouter(nn.Module):
    """Router that considers broader context and task requirements"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_experts = config.get("num_experts", 10)
        
        # Context encoding
        self.context_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Task complexity estimation
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Context-based routing
        self.context_router = nn.Linear(self.hidden_size // 2, self.num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Context-sensitive routing"""
        # Encode context (treating tokens as sequence)
        hidden_states_seq = hidden_states.unsqueeze(1)
        context_output, _ = self.context_encoder(hidden_states_seq)
        context_features = context_output[:, -1, :]  # Take last output
        
        # Estimate task complexity
        complexity = self.complexity_estimator(hidden_states)
        
        # Generate routing based on context
        base_routing = self.context_router(context_features)
        
        # Adjust routing based on complexity (more experts for complex tasks)
        complexity_boost = complexity * 2.0  # Scale factor
        routing_logits = base_routing * complexity_boost
        
        return routing_logits 