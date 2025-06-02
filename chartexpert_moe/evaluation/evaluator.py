import torch

class ChartExpertEvaluator:
    def __init__(self):
        self.metrics_log = {}
        # Mapping metric names to their computation methods
        self.metric_functions = {
            'exact_match': self.compute_exact_match,
            'accuracy_plus': self.compute_accuracy_plus,
            'scrm_score': self.compute_scrm,  # Structural representation metric
            'visual_reasoning_acc': self.compute_visual_reasoning,
            'expert_utilization': self.compute_expert_stats
        }
        print("ChartExpertEvaluator initialized.")

    def compute_exact_match(self, predictions: list, targets: list) -> float:
        """Computes the Exact Match (EM) score."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length.")
        if not predictions: # Handle empty case
            return 0.0 
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(predictions)

    def compute_accuracy_plus(self, predictions: torch.Tensor, targets: torch.Tensor, confidences: torch.Tensor) -> float:
        """
        Enhanced metric penalizing overconfident errors.
        Args:
            predictions (torch.Tensor): Predicted labels/values (e.g., class indices).
            targets (torch.Tensor): Ground truth labels/values.
            confidences (torch.Tensor): Confidence scores for each prediction.
        Returns:
            float: The Accuracy Plus score.
        """
        if not (predictions.shape == targets.shape == confidences.shape):
            raise ValueError("Predictions, targets, and confidences must have the same shape.")
        if predictions.numel() == 0: # Handle empty case
            return 0.0

        correct = (predictions == targets).float()
        # Penalty is applied only to incorrect predictions, scaled by their confidence
        penalty = torch.where(correct == 0, confidences, torch.tensor(0.0, device=confidences.device))
        
        accuracy = correct.mean()
        penalty_score = penalty.mean()
        
        # The paper states: accuracy - 0.1 * penalty.mean()
        # Assuming penalty.mean() is the average confidence of wrong answers.
        # If confidences for correct answers are also included in penalty.mean() (but are 0 due to where clause), it's fine.
        return (accuracy - 0.1 * penalty_score).item()

    def compute_scrm(self, predicted_structures: list, target_structures: list) -> float:
        """Placeholder for Structural Chart Representation Metric (SCRM)."""
        print("SCRM computation is complex and depends on the specific structural representation. This is a placeholder.")
        # Actual implementation would involve comparing graph structures, component alignments, etc.
        if not predicted_structures: return 0.0
        # Example: simple overlap if structures are sets of elements
        # score = sum(1 for ps, ts in zip(predicted_structures, target_structures) if ps == ts) / len(predicted_structures)
        return 0.0 # Placeholder value

    def compute_visual_reasoning(self, predictions: list, targets: list) -> float:
        """Placeholder for Visual Reasoning Accuracy."""
        print("Visual Reasoning Accuracy depends on the specific task format. This is a placeholder.")
        if not predictions: return 0.0
        return self.compute_exact_match(predictions, targets) # Default to EM if no specific logic

    def compute_expert_stats(self, expert_routing_logits: torch.Tensor = None, expert_indices: torch.Tensor = None, num_experts: int = 0) -> dict:
        """
        Computes statistics about expert utilization.
        Args:
            expert_routing_logits (torch.Tensor, optional): Logits from the router. Shape (num_tokens, num_experts).
            expert_indices (torch.Tensor, optional): Indices of chosen expert per token. Shape (num_tokens,).
            num_experts (int, optional): Total number of experts.
        Returns:
            dict: Statistics like frequency per expert.
        """
        stats = {}
        if expert_indices is not None and num_experts > 0:
            if expert_indices.numel() == 0:
                stats['expert_token_distribution'] = [0.0] * num_experts
            else:
                expert_freq = torch.bincount(expert_indices.flatten(), minlength=num_experts).float()
                stats['expert_token_distribution'] = (expert_freq / expert_indices.numel()).tolist()
        else:
            stats['expert_token_distribution'] = "Not computed (expert_indices or num_experts not provided)."
        
        if expert_routing_logits is not None:
            if expert_routing_logits.numel() == 0:
                 stats['avg_router_probabilities'] = [0.0] * num_experts if num_experts > 0 else []
            else:
                avg_probs = torch.softmax(expert_routing_logits, dim=-1).mean(dim=0)
                stats['avg_router_probabilities'] = avg_probs.tolist()
        else:
            stats['avg_router_probabilities'] = "Not computed (expert_routing_logits not provided)."
            
        print("Expert utilization stats computed (placeholder for more detailed analysis).")
        return stats

    def evaluate(self, metric_name: str, **kwargs) -> float:
        """Evaluates a specific metric based on its name."""
        if metric_name not in self.metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}. Available metrics are: {list(self.metric_functions.keys())}")
        
        value = self.metric_functions[metric_name](**kwargs)
        self.metrics_log[metric_name] = value
        print(f"Evaluated {metric_name}: {value}")
        return value

    def get_all_metrics(self) -> dict:
        return self.metrics_log

if __name__ == '__main__':
    evaluator = ChartExpertEvaluator()

    # Example for Exact Match
    em_score = evaluator.evaluate(metric_name='exact_match', predictions=['a', 'b', 'c'], targets=['a', 'b', 'd'])
    # em_score will be 0.666...

    # Example for Accuracy Plus
    # Assume 3 items, 2 correct predictions, 1 incorrect.
    # Incorrect one was predicted with high confidence.
    preds = torch.tensor([0, 1, 0])
    targets = torch.tensor([0, 1, 1]) # Third one is wrong
    confs = torch.tensor([0.9, 0.8, 0.95]) # Confidence for the wrong one is 0.95
    acc_plus_score = evaluator.evaluate(metric_name='accuracy_plus', predictions=preds, targets=targets, confidences=confs)
    # Expected: correct = [1,1,0], penalty = [0,0,0.95]. acc = 2/3. penalty_mean = 0.95/3.
    # score = (2/3) - 0.1 * (0.95/3) = 0.666... - 0.1 * 0.316... = 0.666... - 0.0316... = 0.634...

    # Example for Expert Stats
    # Assume 10 tokens, 3 experts. Router logits and chosen indices.
    num_tokens = 10
    num_exp = 3
    example_logits = torch.randn(num_tokens, num_exp)
    example_indices = torch.randint(0, num_exp, (num_tokens,))
    expert_stats = evaluator.evaluate(metric_name='expert_utilization', 
                                        expert_routing_logits=example_logits, 
                                        expert_indices=example_indices, 
                                        num_experts=num_exp)

    print("All logged metrics:", evaluator.get_all_metrics())
