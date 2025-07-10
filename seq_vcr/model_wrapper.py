import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import sys
import os

# Add src to path to import the original model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ImplicitModel

from .config import SeqVCRConfig
from .regularization import SeqVCRRegularizer
from .pause_tokens import PauseTokenProcessor

def get_penultimate_layer_dim(base_model: ImplicitModel) -> int:
    """
    Get the dimension of the penultimate layer of the base model
    """
    return base_model.base_model.config.hidden_size


class SeqVCRImplicitModel(nn.Module):
    """
    Enhanced ImplicitModel with Sequential Variance-Covariance Regularization (Seq-VCR)
    
    This wrapper adds Seq-VCR capabilities to the existing ImplicitModel without
    modifying the original code, maintaining full backward compatibility.
    """
    
    def __init__(self, base_model: ImplicitModel, seq_vcr_config: SeqVCRConfig):
        super().__init__()
        self.base_model = base_model
        self.config = seq_vcr_config
                
        # Initialize pause token processor
        if seq_vcr_config.enable_pause_tokens:
            self.pause_processor = PauseTokenProcessor(base_model.tokenizer, seq_vcr_config)
            # Resize model embeddings if new tokens were added
            if len(base_model.tokenizer) > base_model.base_model.config.vocab_size:
                base_model.base_model.resize_token_embeddings(len(base_model.tokenizer))
        else:
            self.pause_processor = None
        
        
        def get_model_device(model):
            return next(model.parameters()).device
        
        penultimate_layer_dim = get_penultimate_layer_dim(base_model)        
        device = get_model_device(base_model)
        print(f"Model device: {device}")

        # Initialize Seq-VCR components)
        if seq_vcr_config.enable_seq_vcr:
            self.regularizer = SeqVCRRegularizer(seq_vcr_config, penultimate_layer_dim, device=device)
        else:
            self.regularizer = None


        # Store regularization loss for logging
        self.last_regularization_loss = 0.0
        self.last_regularization_info = {}
    
    def forward(self, input_ids, output_attentions=False, labels=None):
        """
        Enhanced forward pass with optional Seq-VCR regularization
        
        Args:
            input_ids: [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            labels: Optional labels for loss computation
            
        Returns:
            outputs with additional regularization information
        """
        # Process pause tokens if enabled
        if self.pause_processor is not None and labels is not None:
            input_ids, labels = self.pause_processor.add_pause_tokens(input_ids, labels)
        elif self.pause_processor is not None:
            input_ids = self.pause_processor.add_pause_tokens(input_ids)
        
        # Standard forward pass
        if labels is not None:
            outputs = self.base_model.compute_loss(input_ids=input_ids, labels=labels, output_attentions=output_attentions)
        else:
            outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions)
        
        # Add regularization if enabled and in training mode
        if (self.regularizer is not None and 
            self.config.enable_seq_vcr and 
            (not self.config.apply_only_during_training or self.training)):
            
            # Extract hidden representations from final transformer layer
            hidden_states = self.extract_final_layer_representations(input_ids, output_attentions)
            
            # Compute regularization loss
            reg_loss = self.regularizer(hidden_states)
            
            # Store for logging
            self.last_regularization_loss = reg_loss.item()
            self.last_regularization_info = self.regularizer.get_regularization_info(hidden_states)
            
            # Add to total loss if loss exists
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                outputs.loss = outputs.loss + reg_loss
            else:
                # Create loss if it doesn't exist
                outputs.loss = reg_loss
        else:
            self.last_regularization_loss = 0.0
            self.last_regularization_info = {}
        
        return outputs
    
    def extract_final_layer_representations(self, input_ids, output_attentions=False):
        """
        Extract hidden states from final transformer layer before classification head
        
        Args:
            input_ids: [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        # Get transformer outputs with hidden states
        transformer_outputs = self.base_model.base_model.forward(
            input_ids=input_ids, 
            output_hidden_states=True,
            output_attentions=output_attentions
        )
        
        # Get hidden states from final layer
        # transformer_outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Last element is the final layer output
        final_hidden_states = transformer_outputs.hidden_states[-1]
        
        return final_hidden_states
    
    def compute_loss(self, input_ids, labels, output_attentions=False):
        """
        Compute loss including Seq-VCR regularization term
        
        This is the main interface that should be used during training.
        """
        return self.forward(input_ids, output_attentions, labels)
    
    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True):
        """
        Generate text using the base model with pause token handling
        
        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            stop_on_two_eos: Whether to stop on two EOS tokens
            
        Returns:
            generated_sequences: List of generated sequences
        """
        # Add pause tokens if enabled
        if self.pause_processor is not None:
            input_ids = self.pause_processor.add_pause_tokens(input_ids)
        
        # Generate using base model
        generated = self.base_model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            num_beams=num_beams, 
            stop_on_two_eos=stop_on_two_eos
        )
        
        # Remove pause tokens from generated sequences if needed
        if self.pause_processor is not None and len(generated) > 0:
            # Convert list of tensors to single tensor for processing
            if isinstance(generated, list):
                # Pad sequences to same length
                from torch.nn.utils.rnn import pad_sequence
                padded = pad_sequence(
                    [seq.squeeze() for seq in generated], 
                    batch_first=True, 
                    padding_value=self.base_model.tokenizer.eos_token_id
                )
                cleaned = self.pause_processor.remove_pause_tokens(padded)
                # Convert back to list format
                generated = [cleaned[i:i+1] for i in range(cleaned.shape[0])]
        
        return generated
    
    def get_regularization_info(self) -> Dict[str, Any]:
        """Get information about the last regularization computation"""
        return {
            'enabled': self.config.enable_seq_vcr,
            'last_loss': self.last_regularization_loss,
            'pause_tokens_enabled': self.config.enable_pause_tokens,
            'lambda_var': self.config.lambda_var,
            'lambda_cov': self.config.lambda_cov,
            **self.last_regularization_info
        }
    
    def get_pause_token_info(self) -> Dict[str, Any]:
        """Get information about pause token processing"""
        if self.pause_processor is not None:
            return self.pause_processor.get_pause_token_info()
        else:
            return {'enabled': False}
    
    
    @property
    def tokenizer(self):
        """Access to tokenizer"""
        return self.base_model.tokenizer
    
    def save_pretrained(self, save_directory):
        """Save the model and configuration"""
        # Save the base model
        self.base_model.save_pretrained(save_directory)
        
        # Save Seq-VCR configuration
        import json
        from dataclasses import asdict
        config_path = os.path.join(save_directory, 'seq_vcr_config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save regularizer state if it exists
        if self.regularizer is not None:
            reg_path = os.path.join(save_directory, 'seq_vcr_regularizer.pt')
            torch.save(self.regularizer.state_dict(), reg_path)
        
        print(f"Saved Seq-VCR model to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_path, seq_vcr_config=None):
        """Load a Seq-VCR model from a saved directory"""
        # Load base model
        base_model = ImplicitModel.from_pretrained(pretrained_path)
        
        # Load Seq-VCR configuration if not provided
        if seq_vcr_config is None:
            import json
            config_path = os.path.join(pretrained_path, 'seq_vcr_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                seq_vcr_config = SeqVCRConfig(**config_dict)
            else:
                seq_vcr_config = SeqVCRConfig()  # Default config
        
        # Create Seq-VCR model
        model = cls(base_model, seq_vcr_config)
        
        # Load regularizer state if it exists
        if model.regularizer is not None:
            reg_path = os.path.join(pretrained_path, 'seq_vcr_regularizer.pt')
            if os.path.exists(reg_path):
                model.regularizer.load_state_dict(torch.load(reg_path))
        
        return model 