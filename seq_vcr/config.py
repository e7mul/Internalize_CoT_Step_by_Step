from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SeqVCRConfig:
    """Configuration for Sequential Variance-Covariance Regularization (Seq-VCR)"""
    
    # Core regularization parameters
    enable_seq_vcr: bool = True
    lambda_var: float = 1.0      # From paper: λ₁ = 1.0 for multiplication
    lambda_cov: float = 0.004    # From paper: λ₂ = 0.004 for multiplication
    epsilon: float = 0.001       # Numerical stability constant η
    
    projection_dim: int = 2048   # From paper: project to 2048 dims
    
    # Pause token parameters
    enable_pause_tokens: bool = True
    num_pause_tokens: int = 2    # From paper: 2 pause tokens optimal
    pause_token: str = "<pause>"
    pause_start_token: str = "</pause_start>"
    pause_end_token: str = "</pause_end>"
    
    # Training parameters
    apply_only_during_training: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.lambda_var < 0:
            raise ValueError("lambda_var must be non-negative")
        if self.lambda_cov < 0:
            raise ValueError("lambda_cov must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.num_pause_tokens < 0:
            raise ValueError("num_pause_tokens must be non-negative")
    
    @classmethod
    def for_multiplication_task(cls, **kwargs):
        """Default configuration for multiplication tasks as used in the paper"""
        return cls(
            enable_seq_vcr=True,
            lambda_var=1.0,
            lambda_cov=0.004,
            projection_dim=2048,
            **kwargs
        )
    
    @classmethod
    def for_arithmetic_task(cls, **kwargs):
        """Default configuration for other arithmetic tasks as used in the paper"""
        return cls(
            enable_seq_vcr=True,
            lambda_var=0.1,
            lambda_cov=0.5,
            projection_dim=2048,
            **kwargs
        ) 