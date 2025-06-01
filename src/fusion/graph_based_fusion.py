"""
Graph-based fusion for ChartExpert-MoE

This module implements graph-based fusion that integrates structured chart representations
with visual and textual features using graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple, List
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GraphBasedFusion(nn.Module):
    """
    Graph-based fusion for integrating structured chart information
    
    Converts chart elements into a graph structure and fuses it with
    visual and textual representations using GNNs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_gnn_layers = config.get("num_gnn_layers", 3)
        self.graph_hidden_dim = config.get("graph_hidden_dim", 512)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Node feature projection
        self.node_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Edge feature projection (if edge features are available)
        self.edge_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            if i == 0:
                in_dim = self.graph_hidden_dim
            else:
                in_dim = self.graph_hidden_dim * self.num_heads if i > 1 else self.graph_hidden_dim
            
            # Use GAT for better attention-based aggregation
            self.gnn_layers.append(
                GATConv(
                    in_dim,
                    self.graph_hidden_dim,
                    heads=self.num_heads if i < self.num_gnn_layers - 1 else 1,
                    dropout=self.dropout_rate,
                    concat=True if i < self.num_gnn_layers - 1 else False
                )
            )
        
        # Graph-to-sequence attention
        self.graph_to_seq_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.graph_hidden_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        graph_nodes: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply graph-based fusion
        
        Args:
            visual_features: Visual features [batch_size, num_patches, hidden_size]
            text_features: Text features [batch_size, seq_len, hidden_size]
            graph_nodes: Node features from chart-to-graph conversion [num_nodes, hidden_size]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Optional edge features [num_edges, hidden_size]
            batch_idx: Batch indices for nodes [num_nodes]
            
        Returns:
            Fused features [batch_size, seq_len, hidden_size]
        """
        batch_size = visual_features.size(0)
        
        # If no graph structure is provided, use visual features as nodes
        if graph_nodes is None:
            graph_nodes = self._visual_to_graph_nodes(visual_features)
            edge_index = self._create_grid_edges(visual_features.size(1), batch_size)
            batch_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=visual_features.device),
                visual_features.size(1)
            )
        
        # Project node features
        node_features = self.node_projection(graph_nodes)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            node_features = gnn_layer(node_features, edge_index)
            if i < len(self.gnn_layers) - 1:
                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self.dropout_rate, training=self.training)
        
        # Pool graph features per batch
        graph_features = global_mean_pool(node_features, batch_idx)  # [batch_size, graph_hidden_dim]
        
        # Project back to hidden size
        graph_features = self.output_projection(graph_features)  # [batch_size, hidden_size]
        
        # Expand graph features to sequence length
        seq_len = max(visual_features.size(1), text_features.size(1))
        graph_features_expanded = graph_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply graph-to-sequence attention
        attended_features, _ = self.graph_to_seq_attention(
            query=text_features,
            key=graph_features_expanded,
            value=graph_features_expanded
        )
        
        # Three-way fusion with gating
        visual_features_aligned = self._align_features(visual_features, seq_len)
        text_features_aligned = self._align_features(text_features, seq_len)
        
        fusion_input = torch.cat([
            visual_features_aligned,
            text_features_aligned,
            attended_features
        ], dim=-1)
        
        gate_weights = self.fusion_gate(fusion_input)  # [batch_size, seq_len, 3]
        
        # Weighted combination
        fused_features = (
            gate_weights[..., 0:1] * visual_features_aligned +
            gate_weights[..., 1:2] * text_features_aligned +
            gate_weights[..., 2:3] * attended_features
        )
        
        return fused_features
    
    def _visual_to_graph_nodes(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Convert visual patches to graph nodes"""
        batch_size, num_patches, hidden_size = visual_features.shape
        return visual_features.reshape(-1, hidden_size)
    
    def _create_grid_edges(self, num_patches: int, batch_size: int) -> torch.Tensor:
        """Create grid connectivity for visual patches"""
        # Assuming square grid
        grid_size = int(num_patches ** 0.5)
        edges = []
        
        for b in range(batch_size):
            offset = b * num_patches
            for i in range(grid_size):
                for j in range(grid_size):
                    node_idx = i * grid_size + j + offset
                    
                    # Connect to right neighbor
                    if j < grid_size - 1:
                        edges.append([node_idx, node_idx + 1])
                        edges.append([node_idx + 1, node_idx])
                    
                    # Connect to bottom neighbor  
                    if i < grid_size - 1:
                        edges.append([node_idx, node_idx + grid_size])
                        edges.append([node_idx + grid_size, node_idx])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
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