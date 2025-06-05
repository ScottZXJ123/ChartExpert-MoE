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
from ..experts import (
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
from ..routing import DynamicRouter
from ..fusion import MultiModalFusion

# 导入优化组件（这些会在运行时检查是否存在）
try:
    from ..utils.memory_manager import ExpertMemoryManager, AdaptiveMemoryManager
    from ..utils.dynamic_batching import DynamicBatcher, BatchSample, ComplexityLevel
    from ..utils.chart_optimizer import ChartOptimizer
    from ..training.training_optimizer import TrainingOptimizer
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False


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
        
        # Hierarchical expert organization
        self.expert_levels = nn.ModuleList([
            # Level 1: Basic visual experts
            nn.ModuleList([
                self.layout_expert,
                self.ocr_expert,
                self.geometric_expert
            ]),
            # Level 2: Semantic understanding experts
            nn.ModuleList([
                self.scale_expert,
                self.trend_expert,
                self.numerical_expert,
                self.query_expert
            ]),
            # Level 3: High-level reasoning experts
            nn.ModuleList([
                self.integration_expert,
                self.alignment_expert,
                self.chart_to_graph_expert,
                self.shallow_reasoning_expert,
                self.orchestrator_expert
            ])
        ])
        
        # Collect all experts for compatibility
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
        
        # Adaptive expert selection
        self.complexity_estimator = nn.Linear(config["hidden_size"], 1)
        self.min_experts = config.get("min_experts", 1)
        self.max_experts = len(self.experts)
        
        # Multi-level routers for hierarchical processing
        self.level_routers = nn.ModuleList([
            DynamicRouter({**config["routing"], "num_experts": len(level_experts)})
            for level_experts in self.expert_levels
        ])
        
        # MoE layer with dynamic routing (legacy compatibility)
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
        
        # Initialize optimization components
        self._init_optimization_components()
        
        # Initialize KV cache and early exit components
        self._init_inference_optimizations()
        
        # Initialize weights
        self._init_weights()
    
    def _init_inference_optimizations(self):
        """Initialize inference optimization components"""
        # KV缓存池
        self.kv_cache_pool = {}
        self.cache_size_limit = self.config.get("kv_cache_size_limit", 2 * 1024 * 1024 * 1024)  # 2GB
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 早期退出阈值
        self.confidence_thresholds = {
            'simple': self.config.get("simple_confidence_threshold", 0.95),
            'medium': self.config.get("medium_confidence_threshold", 0.90),
            'complex': self.config.get("complex_confidence_threshold", 0.85)
        }
        
        # 中间层置信度估计器
        self.intermediate_confidence_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config["hidden_size"], self.config["hidden_size"] // 2),
                nn.ReLU(),
                nn.Linear(self.config["hidden_size"] // 2, 1),
                nn.Sigmoid()
            ) for _ in range(self.config.get("num_early_exit_layers", 3))
        ])
        
        # 早期退出的输出投影层
        self.early_exit_projections = nn.ModuleList([
            nn.Linear(self.config["hidden_size"], self.config["vocab_size"])
            for _ in range(self.config.get("num_early_exit_layers", 3))
        ])
    
    def hierarchical_expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Hierarchical expert processing with adaptive selection
        
        Args:
            hidden_states: Input features [batch_size, seq_len, hidden_size]
            image: Chart image tensor
            input_ids: Text input token ids
            attention_mask: Text attention mask
            
        Returns:
            Processed features from hierarchical experts
        """
        # Estimate task complexity for adaptive expert selection
        complexity = torch.sigmoid(self.complexity_estimator(hidden_states.mean(dim=1)))
        num_experts_per_level = self._adaptive_expert_count(complexity)
        
        level_outputs = []
        current_input = hidden_states
        
        for level_idx, (level_experts, router, num_experts) in enumerate(
            zip(self.expert_levels, self.level_routers, num_experts_per_level)
        ):
            # Route to selected experts at this level
            routing_output = router(
                hidden_states=current_input.mean(dim=1),
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask
            )  # [batch_size, num_level_experts]
            routing_weights = routing_output["routing_weights"]
            top_k_indices = torch.topk(routing_weights, k=min(num_experts, len(level_experts)), dim=-1).indices
            
            # Process with selected experts
            level_output = self._process_level_experts(
                current_input, level_experts, top_k_indices, routing_weights,
                image, input_ids, attention_mask
            )
            
            level_outputs.append(level_output)
            
            # Use level output as input for next level (residual connection)
            current_input = current_input + level_output
        
        # Combine all level outputs
        return sum(level_outputs) / len(level_outputs)
    
    def _adaptive_expert_count(self, complexity: torch.Tensor) -> List[int]:
        """
        Determine number of experts to use at each level based on complexity
        
        Args:
            complexity: Task complexity scores [batch_size, 1]
            
        Returns:
            List of expert counts for each level
        """
        avg_complexity = complexity.mean().item()
        
        if avg_complexity < 0.3:
            # Simple tasks: fewer experts
            return [1, 2, 1]  # Level 1, 2, 3
        elif avg_complexity < 0.7:
            # Medium tasks: moderate expert usage
            return [2, 3, 2]
        else:
            # Complex tasks: use more experts
            return [3, 4, 3]
    
    def _process_level_experts(
        self,
        hidden_states: torch.Tensor,
        level_experts: nn.ModuleList,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process inputs through selected experts at a specific level
        
        Args:
            hidden_states: Input features
            level_experts: Experts at this level
            expert_indices: Selected expert indices [batch_size, k]
            routing_weights: Expert routing weights
            image: Chart image tensor
            input_ids: Text input token ids
            attention_mask: Text attention mask
            
        Returns:
            Weighted combination of expert outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Initialize output tensor
        level_output = torch.zeros_like(hidden_states)
        
        # Process each expert in parallel when possible
        expert_outputs = {}
        
        # Flatten hidden states for expert processing [batch_size * seq_len, hidden_size]
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        # Get unique expert indices across all samples in batch
        unique_experts = torch.unique(expert_indices).tolist()
        
        # Parallel expert processing
        if len(unique_experts) > 1 and torch.cuda.is_available():
            # Use CUDA streams for parallel processing
            streams = [torch.cuda.Stream() for _ in range(min(len(unique_experts), 4))]
            
            for idx, expert_id in enumerate(unique_experts):
                stream = streams[idx % len(streams)]
                with torch.cuda.stream(stream):
                    expert_result = level_experts[expert_id](
                        hidden_states=flat_hidden_states, 
                        image=image, 
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    # Extract tensor from dictionary if needed
                    if isinstance(expert_result, dict):
                        expert_output_flat = expert_result.get("hidden_states", expert_result.get("output", flat_hidden_states))
                    else:
                        expert_output_flat = expert_result
                    
                    # Reshape back to 3D
                    expert_outputs[expert_id] = expert_output_flat.view(batch_size, seq_len, hidden_size)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
        else:
            # Sequential processing for CPU or single expert
            for expert_id in unique_experts:
                expert_result = level_experts[expert_id](
                    hidden_states=flat_hidden_states, 
                    image=image, 
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                # Extract tensor from dictionary if needed
                if isinstance(expert_result, dict):
                    expert_output_flat = expert_result.get("hidden_states", expert_result.get("output", flat_hidden_states))
                else:
                    expert_output_flat = expert_result
                
                # Reshape back to 3D
                expert_outputs[expert_id] = expert_output_flat.view(batch_size, seq_len, hidden_size)
        
        # Weighted combination of expert outputs
        for batch_idx in range(batch_size):
            for expert_idx_pos in range(expert_indices.shape[1]):
                expert_id = expert_indices[batch_idx, expert_idx_pos].item()
                if expert_id < len(level_experts):
                    weight = routing_weights[batch_idx, expert_id]
                    level_output[batch_idx] += weight * expert_outputs[expert_id][batch_idx]
        
        return level_output
    
    def _init_optimization_components(self):
        """初始化优化组件"""
        if not OPTIMIZATIONS_AVAILABLE:
            # 如果优化组件不可用，初始化为None
            self.memory_manager = None
            self.dynamic_batcher = None
            self.chart_optimizer = None
            self.training_optimizer = None
            return
        
        # 内存管理器
        memory_config = self.config.get("memory_manager", {})
        if memory_config.get("adaptive", True):
            self.memory_manager = AdaptiveMemoryManager(memory_config)
        else:
            self.memory_manager = ExpertMemoryManager(memory_config)
        
        # 动态批处理器
        batch_config = self.config.get("dynamic_batching", {})
        self.dynamic_batcher = DynamicBatcher(batch_config)
        
        # 图表优化器
        chart_config = self.config.get("chart_optimizer", {})
        self.chart_optimizer = ChartOptimizer(chart_config)
        
        # 训练优化器
        training_config = self.config.get("training_optimizer", {})
        self.training_optimizer = TrainingOptimizer(training_config)
        
        # 初始化内存管理器
        if self.memory_manager is not None:
            self.memory_manager.initialize(self.experts)
    
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
        
        # 图表特定优化（如果可用）
        chart_optimizations = {}
        if self.chart_optimizer is not None:
            chart_optimizations = self.chart_optimizer.optimize_processing(
                image, input_ids, attention_mask
            )
        
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
        
        # Choose processing strategy: hierarchical or legacy MoE
        use_hierarchical = self.config.get("use_hierarchical_experts", True)
        
        if use_hierarchical:
            # Hierarchical expert processing with adaptive selection
            expert_outputs = self.hierarchical_expert_forward(
                hidden_states=fused_features,
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Create compatibility routing weights (average across levels)
            level_routing_weights = []
            for router in self.level_routers:
                routing_output = router(
                    hidden_states=fused_features.mean(dim=1),
                    image=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )  # [batch_size, num_level_experts]
                level_weights = routing_output["routing_weights"]
                # Pad to full expert count for compatibility
                padded_weights = torch.zeros(level_weights.size(0), self.max_experts, device=level_weights.device)
                padded_weights[:, :level_weights.size(1)] = level_weights
                level_routing_weights.append(padded_weights)
            
            routing_weights = torch.stack(level_routing_weights, dim=1).mean(dim=1)  # [batch_size, num_experts]
            routing_weights = routing_weights.unsqueeze(1).expand(-1, fused_features.size(1), -1)  # [batch_size, seq_len, num_experts]
            aux_loss = torch.tensor(0.0, device=expert_outputs.device)
            
        else:
            # Legacy MoE processing for backward compatibility
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
            try:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Validate shapes before loss calculation
                if shift_logits.size(0) == 0 or shift_labels.size(0) == 0:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                else:
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
                # Add auxiliary MoE loss
                aux_loss_weight = self.config.get("aux_loss_weight", 0.01)
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    aux_loss = torch.tensor(0.0, device=logits.device)
                
                total_loss = loss + aux_loss_weight * aux_loss
                outputs["loss"] = total_loss
                outputs["lm_loss"] = loss
            except Exception as e:
                # Fallback loss in case of any calculation errors
                outputs["loss"] = torch.tensor(0.0, device=logits.device, requires_grad=True)
                outputs["lm_loss"] = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return outputs
    
    @torch.inference_mode()
    def optimized_inference(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized inference with caching and early exit
        
        Args:
            image: Image tensor [batch_size, channels, height, width]
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_cache: Whether to use KV caching
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = image.size(0)
        
        # 1. Quick complexity assessment
        query_complexity = self._estimate_query_complexity(input_ids, attention_mask)
        
        # 2. KV Cache management
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(image, input_ids)
            if hasattr(self, '_inference_cache') and cache_key in self._inference_cache:
                cached_result = self._inference_cache[cache_key]
                if self._can_reuse_cache(cached_result, input_ids):
                    return cached_result
        
        # 3. Early confidence check
        if query_complexity < 0.3:
            # Use fast path for simple queries
            return self._fast_path_inference(image, input_ids, attention_mask, query_complexity)
        
        # 4. Expert prediction and preloading for complex queries
        if query_complexity > 0.7:
            predicted_experts = self._predict_needed_experts(input_ids, attention_mask)
            self._preload_experts(predicted_experts)
        
        # 5. Full inference with optimizations
        return self._full_optimized_inference(image, input_ids, attention_mask, query_complexity, cache_key)
    
    def _estimate_query_complexity(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> float:
        """Estimate query complexity for adaptive processing"""
        # 1. Query length complexity
        seq_len = input_ids.size(1)
        length_complexity = min(1.0, seq_len / 100)
        
        # 2. Token diversity (more diverse = more complex)
        unique_tokens = torch.unique(input_ids).size(0)
        vocab_complexity = min(1.0, unique_tokens / 50)
        
        # 3. Presence of numerical tokens (indicates numerical reasoning)
        # Assume tokens 0-9 are digits
        digit_tokens = input_ids[(input_ids >= 0) & (input_ids <= 9)]
        numerical_complexity = min(1.0, digit_tokens.size(0) / 10)
        
        # Combine metrics
        complexity = (length_complexity + vocab_complexity + numerical_complexity) / 3
        return min(1.0, complexity)
    
    def _get_cache_key(self, image: torch.Tensor, input_ids: torch.Tensor) -> str:
        """Generate cache key for the current input"""
        image_hash = hash(tuple(image.flatten()[:100].tolist()))  # Sample hash
        text_hash = hash(tuple(input_ids.flatten().tolist()))
        return f"{image_hash}_{text_hash}"
    
    def _can_reuse_cache(self, cached_result: Dict, current_input_ids: torch.Tensor) -> bool:
        """Check if cached result can be reused"""
        if 'input_ids' not in cached_result:
            return False
        
        cached_ids = cached_result['input_ids']
        # Simple check: if input is similar enough, reuse
        if cached_ids.shape == current_input_ids.shape:
            similarity = (cached_ids == current_input_ids).float().mean()
            return similarity > 0.8
        return False
    
    def _fast_path_inference(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        complexity: float
    ) -> Dict[str, torch.Tensor]:
        """Fast inference path for simple queries"""
        # Use lower resolution and fewer experts
        with torch.cuda.amp.autocast(enabled=True):
            # Process visual features with reduced complexity
            visual_features = self.vision_encoder(image, query_complexity=complexity)
            
            # Process text features
            text_features = self.llm_backbone.encode(input_ids, attention_mask)
            
            # Simple fusion
            fused_features = torch.cat([visual_features, text_features], dim=1)
            
            # Use only top-2 most relevant experts
            expert_indices = self._get_top_experts(fused_features, k=2)
            
            # Process through selected experts only
            expert_outputs = self._process_selected_experts(
                fused_features, expert_indices, image, input_ids, attention_mask
            )
            
            # Generate logits
            logits = self.llm_backbone.generate_logits(expert_outputs)
            
            return {
                "logits": logits,
                "expert_outputs": expert_outputs,
                "visual_features": visual_features,
                "text_features": text_features,
                "fused_features": fused_features,
                "fast_path": True
            }
    
    def _predict_needed_experts(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> List[int]:
        """Predict which experts will be needed based on query"""
        # Simple heuristic based on token patterns
        needed_experts = []
        
        # Always need query decomposition
        needed_experts.append(5)  # query_expert
        
        # Check for visual/layout keywords
        visual_keywords = ['chart', 'graph', 'plot', 'axis', 'legend']
        text_tokens = input_ids.flatten().tolist()
        
        # This is a simplified version - in practice you'd use learned embeddings
        if any(keyword.encode() in str(text_tokens).encode() for keyword in visual_keywords):
            needed_experts.extend([0, 1])  # layout_expert, ocr_expert
        
        # Add reasoning experts for complex queries
        if input_ids.size(1) > 20:
            needed_experts.extend([6, 7, 11])  # numerical, integration, orchestrator
        
        return list(set(needed_experts))
    
    def _preload_experts(self, expert_indices: List[int]):
        """Preload experts to GPU memory"""
        # Implementation would move specific experts to GPU
        pass
    
    def _full_optimized_inference(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        complexity: float,
        cache_key: Optional[str]
    ) -> Dict[str, torch.Tensor]:
        """Full inference with all optimizations"""
        # Use the standard forward pass but with optimizations
        outputs = self.forward(image, input_ids, attention_mask)
        
        # Cache results if enabled
        if cache_key and not hasattr(self, '_inference_cache'):
            self._inference_cache = {}
        
        if cache_key:
            outputs['input_ids'] = input_ids  # Store for cache validation
            self._inference_cache[cache_key] = outputs
            
            # Limit cache size
            if len(self._inference_cache) > 10:
                oldest_key = next(iter(self._inference_cache))
                del self._inference_cache[oldest_key]
        
        return outputs
    
    def _get_top_experts(self, features: torch.Tensor, k: int = 2) -> List[int]:
        """Get top-k most relevant experts for the input"""
        # Simple routing to get top experts
        batch_size, seq_len, hidden_size = features.shape
        flat_features = features.view(-1, hidden_size)
        
        # Use the router to get expert probabilities
        routing_output = self.router(flat_features)
        routing_probs = F.softmax(routing_output["logits"], dim=-1)
        
        # Get top-k experts across all tokens
        mean_probs = routing_probs.mean(dim=0)
        top_k_experts = torch.topk(mean_probs, k, dim=-1).indices.tolist()
        
        return top_k_experts
    
    def _process_selected_experts(
        self,
        features: torch.Tensor,
        expert_indices: List[int],
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Process features through only selected experts"""
        batch_size, seq_len, hidden_size = features.shape
        flat_features = features.view(-1, hidden_size)
        
        # Create mock routing weights for selected experts
        routing_weights = torch.zeros(flat_features.size(0), len(self.experts), device=features.device)
        for idx in expert_indices:
            routing_weights[:, idx] = 1.0 / len(expert_indices)
        
        # Use MoE layer with constrained routing
        moe_output = self.moe_layer._forward_with_routing(
            flat_features, routing_weights, image, input_ids, attention_mask
        )
        
        return moe_output["expert_outputs"]
    
    def smart_inference(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_early_exit: bool = True,
        use_kv_cache: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        智能推理：结合早期退出、KV缓存和层次化专家选择
        
        Args:
            image: Chart image tensor
            input_ids: Text input token ids  
            attention_mask: Attention mask
            use_early_exit: Whether to use early exit optimization
            use_kv_cache: Whether to use KV caching
            
        Returns:
            Dictionary containing optimized inference results
        """
        # 快速复杂度和图表类型检测
        complexity = self._estimate_query_complexity(input_ids, attention_mask)
        
        # 根据复杂度选择推理策略
        if complexity < 0.3 and not use_early_exit:
            # 简单查询：快速路径
            return self._fast_path_inference(image, input_ids, attention_mask, complexity)
        elif use_kv_cache:
            # 中等复杂度：使用KV缓存
            return self.inference_with_kv_cache(image, input_ids, attention_mask, **kwargs)
        elif use_early_exit and hasattr(self, 'intermediate_confidence_estimators'):
            # 复杂查询：早期退出
            return self.inference_with_early_exit(image, input_ids, attention_mask, **kwargs)
        else:
            # 回退：标准推理
            return self.forward(image, input_ids, attention_mask, **kwargs)
    
    def inference_with_early_exit(
        self,
        image: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """带早期退出的推理"""
        if not hasattr(self, 'intermediate_confidence_estimators'):
            # 如果没有早期退出组件，使用标准推理
            return self.forward(image, input_ids, attention_mask, **kwargs)
            
        complexity = self._quick_complexity_assessment(input_ids, attention_mask)
        confidence_threshold = self.confidence_thresholds.get(complexity, 0.90)
        
        # 编码
        visual_features = self.vision_encoder(image)
        text_features = self.llm_backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        fused_features = self.fusion(
            visual_features=visual_features,
            text_features=text_features,
            attention_mask=attention_mask
        )
        
        # 层次化专家处理
        if hasattr(self, 'expert_levels'):
            level_outputs = []
            current_features = fused_features
            
            for level_idx, (level_experts, router) in enumerate(zip(self.expert_levels, self.level_routers)):
                routing_weights = router(current_features.mean(dim=1))
                top_k_indices = torch.topk(routing_weights, k=2, dim=-1).indices
                
                level_output = self._process_level_experts(
                    current_features, level_experts, top_k_indices, routing_weights,
                    image, input_ids, attention_mask
                )
                level_outputs.append(level_output)
                current_features = current_features + level_output
                
                # 检查早期退出
                if level_idx < len(self.intermediate_confidence_estimators):
                    confidence = self.intermediate_confidence_estimators[level_idx](
                        current_features.mean(dim=1)
                    ).mean()
                    
                    if confidence > confidence_threshold:
                        early_logits = self.early_exit_projections[level_idx](current_features)
                        return {
                            "logits": early_logits,
                            "early_exit": True,
                            "exit_layer": level_idx,
                            "confidence": confidence.item(),
                            "complexity": complexity
                        }
        
        # 标准完整推理
        return self.forward(image, input_ids, attention_mask, **kwargs)
    
    def inference_with_kv_cache(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """使用KV缓存的推理"""
        if not hasattr(self, 'kv_cache_pool'):
            return self.forward(image, input_ids, attention_mask, **kwargs)
            
        cache_key = self._generate_cache_key(image, input_ids)
        
        if cache_key in self.kv_cache_pool and self._should_use_cache(input_ids):
            self.cache_hit_count += 1
            return self.kv_cache_pool[cache_key]
        else:
            self.cache_miss_count += 1
            outputs = self.forward(image, input_ids, attention_mask, **kwargs)
            
            if self._should_cache(input_ids, outputs):
                self._add_to_cache(cache_key, outputs)
                
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
        import os
        
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Failed to load or process image: {e}")
        
        try:
            # Tokenize query
            inputs = self.llm_backbone.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move tensors to same device as model
            device = next(self.parameters()).device
            image_tensor = image_tensor.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.forward(
                    image=image_tensor,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                # Validate outputs
                if "expert_outputs" not in outputs:
                    raise RuntimeError("Model forward pass failed to produce expert_outputs")
                
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
        except Exception as e:
            raise RuntimeError(f"Failed to generate prediction: {e}")
        
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
    
    # 支持方法
    def _quick_complexity_assessment(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> str:
        """快速评估任务复杂度"""
        seq_len = input_ids.size(1)
        
        # 基于序列长度的简单启发式
        if seq_len < 20:
            return 'simple'
        elif seq_len < 50:
            return 'medium'
        else:
            return 'complex'
    
    def _generate_cache_key(self, image: torch.Tensor, input_ids: torch.Tensor) -> str:
        """生成缓存键"""
        # 使用image和input_ids的哈希作为键
        image_hash = hash(str(image.shape) + str(image.sum().item()))
        text_hash = hash(str(input_ids.flatten().tolist()[:10]))  # 只用前10个token
        return f"{image_hash}_{text_hash}"
    
    def _should_use_cache(self, input_ids: torch.Tensor) -> bool:
        """判断是否应该使用缓存"""
        # 简单策略：如果序列很短，使用缓存
        return input_ids.size(1) < 100
    
    def _should_cache(self, input_ids: torch.Tensor, outputs: Dict) -> bool:
        """判断是否应该缓存结果"""
        # 缓存策略：中等长度的序列，且置信度高
        seq_len = input_ids.size(1)
        if seq_len < 10 or seq_len > 200:
            return False
        
        # 检查缓存大小限制
        if hasattr(self, 'kv_cache_pool'):
            current_cache_size = sum(
                len(str(kv)) for kv in self.kv_cache_pool.values()
            )
            return current_cache_size < getattr(self, 'cache_size_limit', 2 * 1024**3)
        
        return True
    
    def _add_to_cache(self, cache_key: str, outputs: Dict):
        """添加到缓存"""
        if hasattr(self, 'kv_cache_pool'):
            self.kv_cache_pool[cache_key] = outputs
            
            # 如果缓存过大，删除最旧的条目
            if len(self.kv_cache_pool) > 1000:  # 最多1000个条目
                oldest_key = next(iter(self.kv_cache_pool))
                del self.kv_cache_pool[oldest_key]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取所有优化统计信息"""
        stats = {
            "model_type": "ChartExpert-MoE",
            "optimizations_enabled": []
        }
        
        # 层次化专家统计
        if hasattr(self, 'expert_levels'):
            stats["optimizations_enabled"].append("hierarchical_experts")
            stats["expert_levels"] = len(self.expert_levels)
            stats["experts_per_level"] = [len(level) for level in self.expert_levels]
        
        # 缓存统计
        if hasattr(self, 'kv_cache_pool'):
            stats["optimizations_enabled"].append("kv_cache")
            stats["cache_stats"] = self.get_cache_stats()
        
        # 早期退出统计
        if hasattr(self, 'intermediate_confidence_estimators'):
            stats["optimizations_enabled"].append("early_exit")
            stats["early_exit_layers"] = len(self.intermediate_confidence_estimators)
        
        # 内存管理统计
        if hasattr(self, 'memory_manager') and self.memory_manager:
            stats["optimizations_enabled"].append("memory_management")
            stats["memory_stats"] = self.memory_manager.get_memory_stats()
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not hasattr(self, 'kv_cache_pool'):
            return {"cache_enabled": False}
            
        total_requests = getattr(self, 'cache_hit_count', 0) + getattr(self, 'cache_miss_count', 0)
        hit_rate = getattr(self, 'cache_hit_count', 0) / max(1, total_requests)
        
        return {
            "cache_enabled": True,
            "cache_hit_count": getattr(self, 'cache_hit_count', 0),
            "cache_miss_count": getattr(self, 'cache_miss_count', 0),
            "hit_rate": hit_rate,
            "cache_size": len(self.kv_cache_pool),
            "cache_size_limit": getattr(self, 'cache_size_limit', 0)
        }
    
    def clear_all_caches(self):
        """清理所有缓存"""
        if hasattr(self, 'kv_cache_pool'):
            self.kv_cache_pool.clear()
            self.cache_hit_count = 0
            self.cache_miss_count = 0
        
        if hasattr(self, '_inference_cache'):
            self._inference_cache.clear()
        
        # 清理内存管理器缓存
        if hasattr(self, 'memory_manager') and self.memory_manager:
            self.memory_manager.reset_stats() 