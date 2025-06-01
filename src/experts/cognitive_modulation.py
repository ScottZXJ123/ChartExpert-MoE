"""
Cognitive Effort Modulation Expert modules for ChartExpert-MoE

These experts handle cognitive effort scaling and reasoning orchestration including:
- Shallow Reasoning for simple tasks
- Deep Reasoning Orchestrator for complex multi-step reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Tuple

from .base_expert import BaseExpert


class ShallowReasoningExpert(BaseExpert):
    """
    Expert for simple, fast reasoning tasks
    
    Specializes in:
    - Quick data retrieval and direct questions
    - Simple value lookups
    - Basic pattern recognition
    - Fast response generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "shallow_reasoning"
        super().__init__(config)
        
        # Fast processing components
        self.quick_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 8)  # Quick classification categories
        )
        
        # Simple pattern matching
        self.pattern_matcher = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2)
        )
        
        # Fast response generator
        self.response_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Confidence estimator for shallow reasoning
        self.shallow_confidence = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Speed optimization layer
        self.speed_optimizer = nn.Linear(self.hidden_size, self.hidden_size)
    
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build shallow reasoning layers (optimized for speed)"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),  # Faster than GELU
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def _is_simple_task(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Determine if the task is simple enough for shallow reasoning"""
        classification = self.quick_classifier(hidden_states)
        # Simple heuristic: if max probability is high, it's likely a simple task
        max_prob = torch.max(F.softmax(classification, dim=-1), dim=-1)[0]
        return max_prob > 0.8  # Threshold for simplicity
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Shallow reasoning forward pass"""
        # Quick pattern matching
        pattern_features = self.pattern_matcher(hidden_states)
        
        # Speed optimization
        optimized_features = self.speed_optimizer(hidden_states)
        
        # Generate quick response
        response_features = self.response_generator(optimized_features)
        
        # Estimate confidence in shallow reasoning
        confidence = self.shallow_confidence(response_features)
        
        # Weight response by confidence
        confident_response = response_features * confidence
        
        return self.expert_layers(confident_response)


class DeepReasoningOrchestratorExpert(BaseExpert):
    """
    Expert for complex multi-step reasoning and expert orchestration
    
    Specializes in:
    - Complex multi-step reasoning tasks
    - Orchestrating other experts in sequence
    - Managing reasoning chains
    - Dynamic resource allocation
    """
    
    def __init__(self, config: Dict[str, Any]):
        config["expert_type"] = "orchestrator"
        super().__init__(config)
        
        # Complexity analysis
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Multi-step reasoning chain
        self.reasoning_chain = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Expert orchestration components
        self.expert_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 10)  # Select from 10 experts
        )
        
        # Reasoning step predictor
        self.step_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 5)  # Max 5 reasoning steps
        )
        
        # Dynamic resource allocator
        self.resource_allocator = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Low, medium, high resource allocation
            nn.Softmax(dim=-1)
        )
        
        # Reasoning state tracker
        self.state_tracker = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Projector for expert_weights before synthesis
        # Assuming 10 experts as hardcoded in expert_selector and expert_reinforcement
        self.expert_weights_projector = nn.Linear(10, self.hidden_size)

        # Solution synthesizer
        self.solution_synthesizer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # nPMI-inspired expert reinforcement
        self.expert_reinforcement = nn.Parameter(torch.ones(10))  # 10 experts
        
        # Deep reasoning confidence
        self.deep_confidence = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build deep reasoning orchestration layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),  # Larger for complex reasoning
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _analyze_reasoning_complexity(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze the complexity of the reasoning task"""
        complexity_score = self.complexity_analyzer(hidden_states)
        
        # Predict number of reasoning steps needed
        step_logits = self.step_predictor(hidden_states)
        num_steps = torch.argmax(step_logits, dim=-1) + 1  # 1-5 steps
        
        # Determine resource allocation
        resource_allocation = self.resource_allocator(hidden_states)
        
        return {
            "complexity_score": complexity_score,
            "num_steps": num_steps,
            "resource_allocation": resource_allocation,
            "step_logits": step_logits
        }
    
    def _orchestrate_expert_sequence(
        self,
        hidden_states: torch.Tensor,
        num_steps: torch.Tensor
    ) -> torch.Tensor:
        """Orchestrate a sequence of expert activations"""
        batch_size = hidden_states.size(0)
        
        # Initialize reasoning state
        current_state = hidden_states.unsqueeze(1)  # Add sequence dimension
        
        # Track reasoning through multiple steps
        for step in range(int(torch.max(num_steps).item())):
            # Update reasoning state
            reasoning_output, _ = self.state_tracker(current_state)
            current_state = reasoning_output
        
        # Take final state
        final_state = current_state[:, -1, :]  # Last timestep
        
        return final_state
    
    def _select_expert_combination(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Select which experts to activate and their weights"""
        expert_logits = self.expert_selector(hidden_states)
        expert_weights = F.softmax(expert_logits, dim=-1)
        
        # Apply reinforcement learning inspired weighting
        reinforced_weights = expert_weights * self.expert_reinforcement.unsqueeze(0)
        reinforced_weights = F.normalize(reinforced_weights, p=1, dim=-1)
        
        return reinforced_weights
    
    def _synthesize_solution(
        self,
        orchestrated_features: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Synthesize final solution from orchestrated reasoning"""
        # Combine orchestrated features with expert selection context
        projected_expert_weights = self.expert_weights_projector(expert_weights)
        combined_input = torch.cat([
            orchestrated_features,
            projected_expert_weights
        ], dim=-1)
        
        synthesized_solution = self.solution_synthesizer(combined_input)
        return synthesized_solution
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Deep reasoning orchestration forward pass"""
        # Analyze reasoning complexity
        complexity_analysis = self._analyze_reasoning_complexity(hidden_states)
        
        # Orchestrate expert sequence based on complexity
        orchestrated_features = self._orchestrate_expert_sequence(
            hidden_states,
            complexity_analysis["num_steps"]
        )
        
        # Select expert combination
        expert_weights = self._select_expert_combination(hidden_states)
        
        # Synthesize solution
        synthesized_solution = self._synthesize_solution(
            orchestrated_features,
            expert_weights
        )
        
        # Apply deep reasoning layers
        deep_reasoning_output = self.expert_layers(synthesized_solution)
        
        # Estimate confidence in deep reasoning
        confidence = self.deep_confidence(deep_reasoning_output)
        
        # Weight by confidence and complexity
        complexity_weighted = (
            deep_reasoning_output * 
            confidence * 
            complexity_analysis["complexity_score"]
        )
        
        return complexity_weighted
    
    def update_expert_reinforcement(self, expert_idx: int, reward: float):
        """Update expert reinforcement based on performance (RL-inspired)"""
        with torch.no_grad():
            current_weight = self.expert_reinforcement[expert_idx]
            # Simple update rule (could be more sophisticated)
            updated_weight = current_weight + 0.01 * reward
            self.expert_reinforcement[expert_idx] = torch.clamp(updated_weight, 0.1, 2.0)
    
    def get_expert_usage_pattern(self) -> Dict[str, float]:
        """Get current expert usage pattern from reinforcement weights"""
        weights = F.softmax(self.expert_reinforcement, dim=0)
        expert_names = [
            "layout", "ocr", "scale", "geometric", "trend",
            "query", "numerical", "integration", "alignment", "shallow"
        ]
        
        return {
            expert_names[i]: float(weights[i])
            for i in range(len(expert_names))
        } 