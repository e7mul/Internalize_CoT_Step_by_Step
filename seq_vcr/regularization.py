import torch
import torch.nn as nn
from typing import Optional
from .config import SeqVCRConfig


class SeqVCRRegularizer(nn.Module):
    """
    Sequential Variance-Covariance Regularization (Seq-VCR)
    
    Implements the regularization technique described in the paper:
    "SEQ-VCR: Preventing Collapse in Intermediate Transformer Representations for Enhanced Reasoning"
    
    Mathematical formulation:
    L_Seq-VCR = (1/(T×d)) × Σᵢ₌₁ᵀ Σₖ₌₁ᵈ [
        λ₁ × max(0, 1 - √(Cᵢ,ₖ,ₖ + η))  [Variance Term]
        + λ₂ × Σₖ≠ₖ̂(Cᵢ,ₖ,ₖ̂)²           [Covariance Term]
    ]
    """
    
    def __init__(self, config: SeqVCRConfig, hidden_dim: int, device: torch.device):
        super().__init__()
        self.config = config
        self.lambda_var = config.lambda_var
        self.lambda_cov = config.lambda_cov
        self.epsilon = config.epsilon
        

        self.projection_dim = config.projection_dim
        self._init_projection(hidden_dim, device)


    
    def _init_projection(self, input_dim: int, device: torch.device):
        """Initialize projection layer when we first see the input dimension"""
        self.projection = nn.Linear(input_dim, self.projection_dim, bias=False)
        # Initialize with Xavier uniform as suggested for regularization layers
        nn.init.xavier_uniform_(self.projection.weight)
        self.projection.to(device)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute Seq-VCR regularization loss
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] - representations from final transformer layer
            
        Returns:
            regularization_loss: scalar tensor
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply projection if enabled
        hidden_states = self.projection(hidden_states)
        hidden_dim = self.projection_dim
        
        # Compute position-wise covariance matrices
        covariance_matrices = self.compute_position_wise_covariance(hidden_states)
        
        # Compute regularization terms
        variance_term = self.compute_variance_term(covariance_matrices)
        covariance_term = self.compute_covariance_term(covariance_matrices)
        
        # Combine terms according to paper formula: normalized by T×d
        total_loss = (self.lambda_var * variance_term + self.lambda_cov * covariance_term) / (seq_len * hidden_dim)
        
        return total_loss
    
    def compute_position_wise_covariance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix for each sequence position across batch dimension
        
        Implements: Cᵢ = (1/(N-1)) × Σⱼ₌₁ᴺ (Xⱼ,ᵢ - X̄:,ᵢ)(Xⱼ,ᵢ - X̄:,ᵢ)ᵀ
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            covariance_matrices: [seq_len, hidden_dim, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if batch_size == 1:
            # Handle single batch case - return zero covariance
            return torch.zeros(seq_len, hidden_dim, hidden_dim, 
                             device=hidden_states.device, dtype=hidden_states.dtype)
        
        covariance_matrices = []
        
        for i in range(seq_len):
            # Get representations for position i across all samples in batch
            position_reps = hidden_states[:, i, :]  # [batch_size, hidden_dim]
            
            # Compute mean across batch
            mean_rep = position_reps.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            
            # Center the representations
            centered = position_reps - mean_rep  # [batch_size, hidden_dim]
            
            # Compute covariance matrix: (1/(N-1)) × X^T X
            cov_matrix = torch.mm(centered.t(), centered) / (batch_size - 1)
            covariance_matrices.append(cov_matrix)
        
        return torch.stack(covariance_matrices, dim=0)  # [seq_len, hidden_dim, hidden_dim]
    
    def compute_variance_term(self, covariance_matrices: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization term: Σᵢ₌₁ᵀ Σₖ₌₁ᵈ max(0, 1 - √(Cᵢ,ₖ,ₖ + η))
        
        Encourages unit variance in each dimension to prevent collapse
        
        Args:
            covariance_matrices: [seq_len, hidden_dim, hidden_dim]
        Returns:
            variance_loss: scalar tensor
        """
        # Extract diagonal elements (variances)
        variances = torch.diagonal(covariance_matrices, dim1=-2, dim2=-1)  # [seq_len, hidden_dim]
        
        # Apply variance regularization formula: max(0, 1 - √(variance + ε))
        sqrt_var_with_eps = torch.sqrt(variances + self.epsilon)
        variance_term = torch.clamp(1.0 - sqrt_var_with_eps, min=0.0)
        
        # Sum over all positions and dimensions
        return variance_term.sum()
    
    def compute_covariance_term(self, covariance_matrices: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance regularization term: Σᵢ₌₁ᵀ Σₖ₌₁ᵈ Σₖ≠ₖ̂(Cᵢ,ₖ,ₖ̂)²
        
        Penalizes off-diagonal correlations to promote diversity
        
        Args:
            covariance_matrices: [seq_len, hidden_dim, hidden_dim]
        Returns:
            covariance_loss: scalar tensor
        """
        seq_len, hidden_dim, _ = covariance_matrices.shape
        
        # Create mask for off-diagonal elements (k ≠ k̂)
        mask = ~torch.eye(hidden_dim, dtype=torch.bool, device=covariance_matrices.device)
        
        # Extract off-diagonal elements and square them
        off_diagonal_squared = (covariance_matrices * mask.unsqueeze(0)) ** 2
        
        # Sum over all positions and off-diagonal elements
        return off_diagonal_squared.sum()
    
    @torch.no_grad()
    def get_regularization_info(self, hidden_states: torch.Tensor) -> dict:
        """
        Get detailed information about regularization for analysis/debugging
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            info: dict with regularization statistics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        projected_states = self.projection(hidden_states)
        
        # Compute covariance matrices
        covariance_matrices = self.compute_position_wise_covariance(projected_states)
        
        # Get detailed statistics
        variances = torch.diagonal(covariance_matrices, dim1=-2, dim2=-1)
        variance_term = self.compute_variance_term(covariance_matrices)
        covariance_term = self.compute_covariance_term(covariance_matrices)
        total_loss = (self.lambda_var * variance_term + self.lambda_cov * covariance_term) / (seq_len * hidden_dim)
        
        return {
            'total_loss': total_loss.item(),
            'variance_term': variance_term.item(),
            'covariance_term': covariance_term.item(),
            'mean_variance': variances.mean().item(),
            'min_variance': variances.min().item(),
            'max_variance': variances.max().item(),
            'projected_dim': projected_states.shape[-1],
            'original_dim': hidden_states.shape[-1]
        } 