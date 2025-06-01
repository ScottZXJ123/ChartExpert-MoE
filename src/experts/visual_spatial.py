"""
Visual-Spatial & Structural Expert modules for ChartExpert-MoE

These experts handle visual understanding tasks including:
- Layout & Element Detection
- OCR & Text Grounding  
- Scale & Coordinate Interpretation
- Geometric Property Analysis
- Trend & Pattern Perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Tuple
import torchvision.transforms as transforms
from torchvision.models import resnet50

from .base_expert import BaseExpert


class LayoutDetectionExpert(BaseExpert):
    """
    Expert for detecting and understanding chart layout and element positions
    
    Specializes in:
    - Identifying chart components (axes, bars, lines, legends, etc.)
    - Understanding spatial relationships between elements
    - Detecting chart type and structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "layout"
        self.visual_feature_dim = config.get("visual_feature_dim", 2048)
        super().__init__(config)
        self.use_object_detection = config.get("use_object_detection", True)
        self.num_heads = config.get("num_heads", 8)
        self.num_max_elements = config.get("num_max_elements", 5) # Max elements to predict
        self.num_layout_classes = config.get("num_layout_classes", 10) # Number of layout element classes
        
        # For object detection-like capabilities
        if self.use_object_detection:
            self.visual_encoder = self._build_visual_encoder(config)
            # Predict bounding boxes and class logits for a fixed number of elements
            self.bbox_regressor = nn.Linear(self.visual_feature_dim, self.num_max_elements * 4)  # N * (x, y, w, h)
            self.element_classifier = nn.Linear(self.visual_feature_dim, self.num_max_elements * self.num_layout_classes)
        
        # Spatial relationship encoding (processes features from one bounding box for now)
        self.spatial_encoder = nn.Linear(4, self.hidden_size // 4)  # For spatial features from one bbox
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build layout detection specific layers"""
        layers = [
            nn.Linear(self.hidden_size + self.visual_feature_dim + self.hidden_size // 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
        ]
        return nn.Sequential(*layers)
    
    def _build_visual_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build visual encoder for layout detection"""
        # Use pretrained ResNet as visual backbone
        backbone = resnet50(weights='IMAGENET1K_V1')  # Updated to use weights parameter
        # Remove final classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add adaptive pooling and projection
        # This results in a single feature vector per image
        return nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1, 1)), # Changed to (1,1) to be explicit about global features
            nn.Flatten(),
            nn.Linear(2048, self.visual_feature_dim), # ResNet50 block output is 2048
            nn.ReLU()
        )
    
    def _extract_spatial_features(self, pred_bboxes_per_image: Optional[torch.Tensor], flat_batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract spatial features from predicted bounding boxes.
        For now, uses the first predicted bounding box.
        pred_bboxes_per_image: Shape [actual_batch_size, num_max_elements, 4]
        """
        if pred_bboxes_per_image is None:
            # Return zeros if no bounding boxes are available (e.g., no image)
            return torch.zeros(flat_batch_size, 4, device=device) # Outputting raw box features
        
        actual_batch_size = pred_bboxes_per_image.size(0)
        
        # Take the first predicted bounding box for simplicity
        # Ensure it's detached if it comes directly from a layer that requires grad
        spatial_features_source = pred_bboxes_per_image[:, 0, :].detach()  # [actual_batch_size, 4]
        
        # Expand to match flattened batch size if needed
        if flat_batch_size > actual_batch_size:
            tokens_per_batch = flat_batch_size // actual_batch_size
            # Ensure tokens_per_batch is an integer, handle cases where it might not be
            if flat_batch_size % actual_batch_size != 0:
                # This case should ideally not happen if hidden_states comes from image + text tokens
                # Fallback: just repeat the first batch's features, or average, or error out
                # For now, we'll assume it divides cleanly based on typical model structure
                pass # Or log a warning

            expanded_features = spatial_features_source.unsqueeze(1).repeat(1, tokens_per_batch, 1)
            final_spatial_features = expanded_features.reshape(flat_batch_size, 4)
        elif flat_batch_size == actual_batch_size: # often B*1 for CLS token like features
            final_spatial_features = spatial_features_source
        else: # flat_batch_size < actual_batch_size - this shouldn't happen
            # Fallback or error
            final_spatial_features = torch.zeros(flat_batch_size, 4, device=device)


        return final_spatial_features
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Layout detection forward pass"""
        flat_batch_size = hidden_states.size(0)  # This is batch_size * seq_len (or just batch_size if global)
        actual_batch_size = image.size(0) if image is not None else (flat_batch_size if hidden_states.size(0) == attention_mask.size(0) else 1) # Heuristic for actual_batch_size
        
        pred_bboxes_for_spatial_features: Optional[torch.Tensor] = None

        # Extract visual features if image is provided
        if image is not None and self.use_object_detection:
            visual_features_global = self.visual_encoder(image)  # [actual_batch_size, visual_feature_dim]
            
            # Predict bounding boxes and element classes from global visual features
            pred_bboxes_flat = self.bbox_regressor(visual_features_global) # [actual_batch_size, num_max_elements * 4]
            pred_bboxes_for_spatial_features = pred_bboxes_flat.view(actual_batch_size, self.num_max_elements, 4)
            
            # pred_elem_logits_flat = self.element_classifier(visual_features_global)
            # pred_elem_logits = pred_elem_logits_flat.view(actual_batch_size, self.num_max_elements, self.num_layout_classes)
            # These are not directly used in combined_features yet, but are predicted.

            # Expand global visual features to match flattened batch size (e.g., token sequence)
            if flat_batch_size > actual_batch_size:
                tokens_per_batch = flat_batch_size // actual_batch_size
                # Ensure clean division
                if flat_batch_size % actual_batch_size != 0 and attention_mask is not None: # If attention_mask suggests seq_len > 1
                     # This logic assumes hidden_states might be (B*S, D) and image features are (B, D_vis)
                     # We need to repeat image features S times for each item in batch B.
                     if attention_mask.size(0) == actual_batch_size: # attention_mask is (B,S)
                         seq_len = attention_mask.size(1)
                         if flat_batch_size == actual_batch_size * seq_len : # Check consistency
                             tokens_per_batch = seq_len
                         else: # Mismatch, default to no expansion or error
                             tokens_per_batch = 1 # Avoids error, but might be wrong
                     else: # Cannot infer seq_len safely
                         tokens_per_batch = 1


                expanded_visual_features = visual_features_global.unsqueeze(1).repeat(1, tokens_per_batch, 1)
                visual_features_to_combine = expanded_visual_features.reshape(flat_batch_size, self.visual_feature_dim)
            elif flat_batch_size == actual_batch_size:
                 visual_features_to_combine = visual_features_global
            else: # Fallback if flat_batch_size is smaller (should not happen with typical MoE inputs)
                visual_features_to_combine = torch.zeros(flat_batch_size, self.visual_feature_dim, device=hidden_states.device)

        else:
            visual_features_to_combine = torch.zeros(flat_batch_size, self.visual_feature_dim, device=hidden_states.device)
        
        # Extract spatial features using the predicted bboxes
        raw_spatial_features = self._extract_spatial_features(pred_bboxes_for_spatial_features, flat_batch_size, hidden_states.device)
        spatial_encoded = self.spatial_encoder(raw_spatial_features) # Input: [flat_batch_size, 4], Output: [flat_batch_size, H//4]
        
        # Combine all features
        combined_features = torch.cat([
            hidden_states,
            visual_features_to_combine,
            spatial_encoded
        ], dim=-1)
        
        return self.expert_layers(combined_features)


class OCRGroundingExpert(BaseExpert):
    """
    Expert for OCR text extraction and grounding in chart context
    
    Specializes in:
    - High-precision text extraction from charts
    - Associating text with visual elements and positions
    - Understanding text hierarchy and relationships
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "ocr"
        super().__init__(config)
        self.num_ocr_detections = config.get("num_ocr_detections", 3) # Number of mock OCR detections
        
        # OCR-specific components
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            batch_first=True
        )
        
        # Position encoding for text grounding: processes features from N detected text boxes
        self.position_encoder = nn.Linear(self.num_ocr_detections * 4, self.hidden_size // 4)  # N * (x, y, w, h)
        
        # Text-visual alignment
        self.text_visual_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build OCR and grounding specific layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size // 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _extract_text_positions(self, image: Optional[torch.Tensor], input_ids: Optional[torch.Tensor], flat_batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract text bounding boxes and positions.
        For now, generates a fixed number of mock bounding boxes per image.
        Output shape: [flat_batch_size, num_ocr_detections * 4]
        """
        if image is None:
            # Return zeros if no image is available
            return torch.zeros(flat_batch_size, self.num_ocr_detections * 4, device=device)
        
        actual_batch_size = image.size(0)
        
        # Generate mock bounding boxes for N OCR detections per image
        # Shape: [actual_batch_size, num_ocr_detections, 4]
        mock_bboxes_per_image = torch.rand(actual_batch_size, self.num_ocr_detections, 4, device=device)
        
        # Flatten the box features: [actual_batch_size, num_ocr_detections * 4]
        source_features = mock_bboxes_per_image.view(actual_batch_size, -1)
        
        # Expand to match flattened batch size (e.g., token sequence length)
        if flat_batch_size > actual_batch_size:
            tokens_per_batch = flat_batch_size // actual_batch_size
            if flat_batch_size % actual_batch_size != 0:
                # Handle potential mismatch if clean division is not possible
                # This case implies complex batching, for now assume clean division or B*1 tokens
                pass # Or log a warning
            expanded_features = source_features.unsqueeze(1).repeat(1, tokens_per_batch, 1)
            final_features = expanded_features.reshape(flat_batch_size, self.num_ocr_detections * 4)
        elif flat_batch_size == actual_batch_size:
            final_features = source_features
        else: # flat_batch_size < actual_batch_size - should not happen
            final_features = torch.zeros(flat_batch_size, self.num_ocr_detections * 4, device=device)
        
        return final_features
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """OCR grounding forward pass"""
        flat_batch_size = hidden_states.size(0)
        
        # Extract text positions from image (mocked for now)
        text_positions = self._extract_text_positions(image, input_ids, flat_batch_size, hidden_states.device)
        
        # Encode positions
        position_encoded = self.position_encoder(text_positions) # Output: [flat_batch_size, H//4]
        
        # Apply text attention for better text understanding
        hidden_states_seq = hidden_states.unsqueeze(1)  # Add sequence dimension
        attended_text, _ = self.text_attention(
            hidden_states_seq, hidden_states_seq, hidden_states_seq
        )
        attended_text = attended_text.squeeze(1)  # Remove sequence dimension
        
        # Combine text features with position information
        combined_features = torch.cat([attended_text, position_encoded], dim=-1)
        
        return self.expert_layers(combined_features)


class ScaleInterpretationExpert(BaseExpert):
    """
    Expert for understanding axis scales and coordinate systems
    
    Specializes in:
    - Reading axis labels and tick marks
    - Understanding scale types (linear, logarithmic, categorical)
    - Mapping visual positions to data values
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "scale"
        super().__init__(config)
        self.num_scale_features = config.get("num_scale_features", 8) # e.g., x_min, x_max, x_ticks, x_type, y_min, y_max, y_ticks, y_type
        
        # Scale type classification
        self.scale_type_classifier = nn.Linear(self.hidden_size, 4)  # linear, log, categorical, time (per axis or global?)
        
        # Value mapping components
        # Takes hidden_states (processed) + extracted scale features
        self.coordinate_mapper = nn.Linear(self.hidden_size + self.num_scale_features, self.hidden_size)
        
        # Numerical reasoning for scale interpretation
        self.numerical_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build scale interpretation layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _extract_scale_features(self, image: Optional[torch.Tensor], flat_batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract scale-related features from image.
        For now, generates mock features for X and Y axes: [min, max, num_ticks, type_code] each.
        Output shape: [flat_batch_size, num_scale_features]
        """
        if image is None:
            return torch.zeros(flat_batch_size, self.num_scale_features, device=device)
        
        actual_batch_size = image.size(0)
        
        # Generate mock features: e.g., x_min, x_max, x_ticks, x_type_code, y_min, y_max, y_ticks, y_type_code
        # Assuming self.num_scale_features is 8 for this example structure.
        # 실제 구현에서는 이러한 값들을 이미지에서 추출해야 합니다.
        # (In a real implementation, these values would be extracted from the image)
        mock_features_per_image = torch.rand(actual_batch_size, self.num_scale_features, device=device)
        # Example: Normalize or set realistic ranges for mock data
        # mock_features_per_image[:, 0] = torch.rand(actual_batch_size, device=device) * 10 # x_min
        # mock_features_per_image[:, 1] = mock_features_per_image[:, 0] + torch.rand(actual_batch_size, device=device) * 100 # x_max
        # mock_features_per_image[:, 2] = torch.randint(3, 10, (actual_batch_size,), device=device).float() # x_ticks
        # mock_features_per_image[:, 3] = torch.randint(0, 3, (actual_batch_size,), device=device).float() # x_type (0:lin, 1:log, 2:cat)
        # Similar for y-axis for features 4-7 if num_scale_features is 8

        # Expand to match flattened batch size
        if flat_batch_size > actual_batch_size:
            tokens_per_batch = flat_batch_size // actual_batch_size
            if flat_batch_size % actual_batch_size != 0:
                pass # Or log a warning
            expanded_features = mock_features_per_image.unsqueeze(1).repeat(1, tokens_per_batch, 1)
            final_features = expanded_features.reshape(flat_batch_size, self.num_scale_features)
        elif flat_batch_size == actual_batch_size:
            final_features = mock_features_per_image
        else: # flat_batch_size < actual_batch_size
            final_features = torch.zeros(flat_batch_size, self.num_scale_features, device=device)
        
        return final_features
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Scale interpretation forward pass"""
        flat_batch_size = hidden_states.size(0)
        
        # Extract scale features (mocked for now)
        scale_features = self._extract_scale_features(image, flat_batch_size, hidden_states.device)
        
        # Process numerical aspects from hidden_states
        numerical_features = self.numerical_processor(hidden_states)
        
        # Optionally, predict scale type (not directly used by coordinate_mapper in this simplified version yet)
        # predicted_scale_type_logits = self.scale_type_classifier(numerical_features) 
        
        # Combine numerical features (from query/context) with extracted scale features
        combined_input = torch.cat([numerical_features, scale_features], dim=-1)
        coordinate_mapped_features = self.coordinate_mapper(combined_input)
        
        return self.expert_layers(coordinate_mapped_features)


class GeometricPropertyExpert(BaseExpert):
    """
    Expert for analyzing geometric properties of chart elements
    
    Specializes in:
    - Measuring bar heights, line slopes, angles
    - Comparing sizes, areas, distances
    - Extracting quantitative visual features
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "geometric"
        super().__init__(config)
        
        # Geometric feature processors
        self.shape_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size + 8, self.hidden_size),  # +8 for geometric features
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Comparison operations
        self.comparison_layer = nn.Linear(self.hidden_size, self.hidden_size)
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build geometric analysis layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _extract_geometric_features(self, image: torch.Tensor, flat_batch_size: int) -> torch.Tensor:
        """Extract geometric properties from chart image"""
        if image is None:
            return torch.zeros(flat_batch_size, 8, device=self.shape_analyzer[0].weight.device)
        
        # Extract features for actual batch
        actual_batch_size = image.size(0)
        geometric_features = torch.randn(actual_batch_size, 8, device=image.device)
        
        # Expand to match flattened batch size
        if flat_batch_size > actual_batch_size:
            tokens_per_batch = flat_batch_size // actual_batch_size
            geometric_features = geometric_features.unsqueeze(1).repeat(1, tokens_per_batch, 1)
            geometric_features = geometric_features.view(flat_batch_size, 8)
        
        return geometric_features
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Geometric analysis forward pass"""
        flat_batch_size = hidden_states.size(0)
        
        # Extract geometric features
        geometric_features = self._extract_geometric_features(image, flat_batch_size)
        
        # Analyze shapes and properties
        shape_input = torch.cat([hidden_states, geometric_features], dim=-1)
        shape_analyzed = self.shape_analyzer(shape_input)
        
        # Apply comparison operations
        comparison_enhanced = self.comparison_layer(shape_analyzed)
        
        return self.expert_layers(comparison_enhanced)


class TrendPatternExpert(BaseExpert):
    """
    Expert for identifying trends and patterns in chart data
    
    Specializes in:
    - Detecting upward/downward trends
    - Identifying patterns and correlations  
    - Recognizing outliers and anomalies
    - Understanding temporal sequences
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "trend"
        super().__init__(config)
        
        # Trend analysis components
        self.sequence_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Pattern recognition
        self.pattern_detector = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # Trend classification
        self.trend_classifier = nn.Linear(self.hidden_size, 5)  # up, down, flat, cyclic, irregular
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build trend analysis layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _extract_temporal_features(self, image: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract temporal/sequential features for trend analysis"""
        if image is None:
            # Use hidden states as sequence
            sequence = hidden_states.unsqueeze(1)  # Add sequence dimension
        else:
            # In practice, would extract time series from chart
            flat_batch_size = hidden_states.size(0)
            sequence = hidden_states.unsqueeze(1).repeat(1, 10, 1)  # Create dummy sequence
        
        return sequence
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Trend pattern analysis forward pass"""
        # Extract temporal features
        sequence = self._extract_temporal_features(image, hidden_states)
        
        # LSTM for sequence modeling
        lstm_output, _ = self.sequence_encoder(sequence)
        # Take the last output
        lstm_features = lstm_output[:, -1, :]  # [batch_size, hidden_size]
        
        # Pattern detection with conv1d
        # Reshape for conv1d: [batch_size, hidden_size, seq_len]
        conv_input = sequence.transpose(1, 2)
        pattern_features = self.pattern_detector(conv_input)
        # Global average pooling
        pattern_features = torch.mean(pattern_features, dim=2)  # [batch_size, hidden_size]
        
        # Combine LSTM and pattern features
        combined_features = lstm_features + pattern_features
        
        return self.expert_layers(combined_features) 