import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartExpertRouter(nn.Module):
    def __init__(self, d_model: int, num_chart_types: int = 6, experts_per_type: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_chart_types = num_chart_types
        self.experts_per_type = experts_per_type

        # Chart type prediction (bar, line, pie, scatter, complex, other)
        self.chart_type_predictor = nn.Linear(d_model, num_chart_types)
        
        # Context enhancement for query-aware routing
        # Input dimension will be d_model for visual_features.mean, d_model for text_features.mean, and d_model for metadata (assuming metadata is also projected to d_model)
        # So, d_model * 3 might be too large if metadata is not d_model. The paper says d_model * 3 for (visual + text + metadata)
        # For the forward pass, it concatenates visual_features.mean(dim=1), text_features.mean(dim=1), and metadata.
        # Let's assume metadata is also of size d_model for now.
        self.context_gate_input_dim = d_model * 3 # For chart_representation used in chart_type_predictor
        self.chart_type_predictor = nn.Linear(self.context_gate_input_dim, num_chart_types)

        # The paper mentions: self.context_gate = nn.Sequential(nn.Linear(d_model * 3, d_model), ...)
        # And in forward: context_features = self.context_gate(torch.cat([visual_features, text_features, metadata], dim=-1))
        # This implies that visual_features, text_features, and metadata are concatenated along the last dimension.
        # If visual_features is (B, N_v, D), text_features is (B, N_t, D), metadata is (B, D_m)
        # This concatenation isn't straightforward unless N_v = N_t = 1 or features are flattened/pooled first.
        # Given the paper's forward pass for chart_type_predictor (uses .mean(dim=1)), let's assume for context_gate input, features are also pooled or represent global features.
        # If visual_features, text_features, metadata are all (B, d_model) after some processing (like mean pooling for sequences):
        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # visual_pooled + text_pooled + metadata_proj
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.chart_types_list = ['bar', 'line', 'pie', 'scatter', 'complex', 'other']
        if len(self.chart_types_list) != num_chart_types:
            raise ValueError(f"Length of chart_types_list must match num_chart_types ({num_chart_types})")

        self.expert_routers = nn.ModuleDict({
            chart_type: nn.Linear(d_model, experts_per_type)
            for chart_type in self.chart_types_list
        })
        print(f"ChartExpertRouter initialized: d_model={d_model}, num_chart_types={num_chart_types}, experts_per_type={experts_per_type}")

    def forward(self, visual_features_pooled: torch.Tensor, text_features_pooled: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Assuming visual_features_pooled, text_features_pooled, and metadata are all of shape (batch_size, d_model)
        # visual_features_pooled: e.g., output of MoonViTEncoder, mean pooled: (B, llm_dim), then projected to d_model if llm_dim != d_model
        # text_features_pooled: e.g., LLM text embeddings, mean pooled: (B, llm_dim), then projected to d_model
        # metadata: (B, d_model) - assumed to be pre-processed to this shape

        if not (visual_features_pooled.ndim == 2 and text_features_pooled.ndim == 2 and metadata.ndim == 2):
            raise ValueError("Pooled visual features, text features, and metadata must be 2D tensors (batch_size, d_model).")
        if not (visual_features_pooled.shape[1] == self.d_model and \
                text_features_pooled.shape[1] == self.d_model and \
                metadata.shape[1] == self.d_model):
            raise ValueError(f"All pooled features and metadata must have dimension d_model ({self.d_model}).")

        # Chart type routing
        chart_representation = torch.cat([
            visual_features_pooled,
            text_features_pooled,
            metadata
        ], dim=-1) # Shape: (batch_size, d_model * 3)
        
        chart_type_logits = self.chart_type_predictor(chart_representation) # Shape: (batch_size, num_chart_types)
        # chart_type_probabilities = F.softmax(chart_type_logits, dim=-1)
        # predicted_chart_type_indices = chart_type_probabilities.argmax(dim=-1) # Shape: (batch_size)

        # Context-enhanced expert selection
        # The paper implies visual_features, text_features, metadata are concatenated for context_gate.
        # This could mean the raw token sequences if context_gate handles sequences, or pooled features.
        # The ChartExpertRouter class diagram shows `d_model` as input to expert_routers, which comes from context_gate.
        # Let's stick to the `d_model*3` input for context_gate, using the same pooled features as for chart_type_predictor.
        context_gate_input = chart_representation # Reusing chart_representation as it's already (B, d_model * 3)
        context_features = self.context_gate(context_gate_input) # Shape: (batch_size, d_model)
        
        # Route to appropriate experts based on chart type
        # This part requires knowing the predicted chart type for each item in the batch to select the correct router.
        # The current structure of expert_routers is a ModuleDict, so we'd iterate or use the predicted indices.
        # For batch processing, it's more efficient to pass all context_features through all routers and then select.
        # However, the spec implies dynamic selection: self.expert_routers[chart_type](context_features)
        # This line would not work directly in a batched forward pass if chart_type varies per item.
        # Let's assume for now this forward pass is for a single item or a batch where all items are of the same predicted chart type.
        # A more robust batched implementation would compute all possible expert logits and then select/mask.
        
        # Simplified: For demonstration, let's assume we use the argmax of chart_type_logits for the *batch* 
        # (or that this router is called per item, or chart_type is passed in).
        # This is a simplification; true batched routing with dynamic expert selection is more complex.
        
        # Get the predicted chart type string for routing (simplified: taking the first item's prediction for the whole batch for ModuleDict access)
        # This is NOT ideal for batched inference if chart types differ in the batch.
        # A real implementation would need to handle this heterogeneity, perhaps by processing one by one or having specific batching strategies.
        predicted_chart_type_idx_for_batch = chart_type_logits.argmax(dim=-1)[0].item()
        selected_chart_type_str = self.chart_types_list[predicted_chart_type_idx_for_batch]
        
        expert_logits = self.expert_routers[selected_chart_type_str](context_features) # Shape: (batch_size, experts_per_type)
        
        return expert_logits, chart_type_logits


class HierarchicalChartRouter(nn.Module):
    def __init__(self, num_experts: int, routing_strategy: str = 'context_aware', load_balance: bool = True, d_model: int = 768):
        super().__init__()
        self.num_experts = num_experts
        self.routing_strategy = routing_strategy
        self.load_balance = load_balance
        self.d_model = d_model
        # This is a placeholder. The actual implementation of a hierarchical router
        # would be more complex, potentially involving multiple levels of routing decisions
        # and integrating with the 12 specific experts.
        # For now, let's assume it's a single layer router to the `num_experts`.
        self.router_layer = nn.Linear(self.d_model, self.num_experts)
        print(f"Placeholder HierarchicalChartRouter initialized: num_experts={num_experts}, strategy='{routing_strategy}', load_balance={load_balance}, d_model={d_model}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (batch_size, d_model) - input features for routing
        # This is a highly simplified placeholder.
        # A true hierarchical router might take multiple types of features, 
        # make sequential decisions, etc.
        if features.shape[-1] != self.d_model:
            raise ValueError(f"Input feature dim {features.shape[-1]} does not match HierarchicalChartRouter d_model {self.d_model}")
        
        expert_logits = self.router_layer(features) # (batch_size, num_experts)
        return expert_logits
