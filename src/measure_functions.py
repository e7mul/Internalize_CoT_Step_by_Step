import torch

def compute_entropy(attn_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of an attention probability distribution
    attn_probs: [batch, heads, seq_len, seq_len]
    return: [batch, heads]
    """
    epsilon = 1e-9
    attn_probs_stable = torch.clamp(attn_probs, min=epsilon)
    entropy = -torch.sum(attn_probs_stable * torch.log(attn_probs_stable), dim=-1).mean(dim=-1)
    assert entropy.shape == (attn_probs.shape[0], attn_probs.shape[1]), "Entropy shape mismatch"
    return entropy


def compute_sink_rate(attn_map: torch.Tensor, threshold: float = 0.3) -> float:
    """Compute sink rate for an attention map"""
    return torch.mean(attn_map[:, :, :, 0],dim=-1) > threshold


def compute_attn_probs(attn_map: torch.Tensor, head_dim: int):
    """
    attn_map: [batch, heads, seq_len, seq_len]
    return: [batch, heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, _ = attn_map.shape
    
    # Apply scaling using actual head dimension
    scaled_attn = attn_map / torch.sqrt(torch.tensor(head_dim, dtype=attn_map.dtype))
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1,seq_len, seq_len)
    masked_attn = scaled_attn.masked_fill(causal_mask == 0, float('-inf'))
    
    # Apply softmax to get proper probability distribution
    attn_probs = torch.softmax(masked_attn.float(), dim=-1)
    assert attn_probs.shape == (batch_size, num_heads, seq_len, seq_len), "Attention probabilities shape mismatch"
    return attn_probs