import torch
import torch.nn.functional as F

def compute_moe_losses(router_logits: torch.Tensor, expert_indices: torch.Tensor, num_experts: int, 
                         aux_loss_weight: float = 0.01, z_loss_weight: float = 0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes MoE specific losses: load balancing loss and router z-loss.

    Args:
        router_logits (torch.Tensor): Output logits from the router. Shape (batch_size * seq_len, num_experts) or (num_tokens, num_experts).
        expert_indices (torch.Tensor): Tensor indicating which expert each token was routed to. 
                                     Assumed to be flattened. Shape (batch_size * seq_len) or (num_tokens).
                                     These are the indices of the chosen expert for each token.
        num_experts (int): Total number of experts.
        aux_loss_weight (float): Weight for the load balancing loss.
        z_loss_weight (float): Weight for the router z-loss.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - total_aux_loss: Weighted sum of load_balance_loss and z_loss.
            - load_balance_loss: The calculated load balancing loss.
            - z_loss: The calculated router z-loss.
    """
    if router_logits.ndim != 2 or expert_indices.ndim != 1:
        raise ValueError("router_logits must be 2D (tokens, num_experts) and expert_indices must be 1D (tokens).")
    if router_logits.shape[0] != expert_indices.shape[0]:
        raise ValueError("Number of tokens in router_logits and expert_indices must match.")
    if router_logits.shape[1] != num_experts:
        raise ValueError(f"Router_logits second dim ({router_logits.shape[1]}) must match num_experts ({num_experts}).")

    # Load balancing loss
    # Calculate frequency of tokens per expert
    expert_freq = torch.bincount(expert_indices, minlength=num_experts).float()
    expert_freq = expert_freq / expert_indices.numel() # Normalize to probabilities
    
    # Calculate importance (average router probability for each expert)
    # router_probs = F.softmax(router_logits, dim=-1)
    # importance = router_probs.mean(dim=0)
    # The paper's formula: num_experts * sum(expert_freq * importance)
    # This form is common in MoE literature. Let's use the importance as mean of softmax probabilities.
    importance = F.softmax(router_logits, dim=-1).mean(dim=0)
    
    # CV^2 based load balancing loss (another common variant, similar effect)
    # load_balance_loss = num_experts * torch.sum(expert_freq * importance)
    # A simpler and often effective version is to encourage router_logits to be uniform for each token across experts if they were all chosen equally.
    # Or, to make sure each expert gets a similar number of tokens (expert_freq should be uniform).
    
    # Using the formula from the document: num_experts * sum(P_i * f_i)
    # P_i is fraction of tokens routed to expert i (expert_freq)
    # f_i is fraction of router probability mass assigned to expert_i (importance)
    load_balance_loss = num_experts * torch.sum(expert_freq * importance)
    
    # Router z-loss for stability (encourages router logits to be small)
    # This helps prevent large logits which can cause instability during training.
    z_loss = torch.sum(router_logits ** 2) / router_logits.numel()
    # An alternative z-loss (logsumexp trick for numerical stability, common in Switch Transformers):
    # z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() / router_logits.numel()
    # Sticking to the simpler one from the doc for now.
    
    total_aux_loss = (aux_loss_weight * load_balance_loss) + (z_loss_weight * z_loss)
    
    return total_aux_loss, load_balance_loss, z_loss
