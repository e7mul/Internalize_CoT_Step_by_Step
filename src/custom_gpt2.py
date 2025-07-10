import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import math

class TemperatureScaledGPT2Attention(GPT2Attention):
    """
    GPT-2 attention with learnable/fixed temperature scaling per head.
    
    Args:
        config: GPT-2 configuration
        is_cross_attention: Whether this is cross-attention
        layer_idx: Layer index for this attention module
        temperature_init_value: Initial value for temperature parameters (default: 1.0)
        temperature_learnable: Whether temperature parameters are learnable (default: True)
    """
    
    def __init__(self, config, is_cross_attention=False, layer_idx=None, 
                 temperature_init_value=1.0, temperature_learnable=True):
        super().__init__(config, is_cross_attention, layer_idx)
        
        self.num_heads = config.num_attention_heads
        self.temperature_init_value = temperature_init_value
        self.temperature_learnable = temperature_learnable
        
        # Initialize temperature parameters
        # Convert init value to log space for numerical stability
        init_log_temp = math.log(temperature_init_value)
        
        if temperature_learnable:
            # Learnable temperature parameters - one per attention head
            self.temperature_logits = nn.Parameter(
                torch.full((self.num_heads,), init_log_temp, dtype=torch.float32)
            )   
        else:
            # Fixed temperature parameters - registered as buffer (not learnable)
            self.register_buffer(
                'temperature_logits',
                torch.full((self.num_heads,), init_log_temp, dtype=torch.float32)
            )
        
        print(f"Layer {layer_idx}: Temperature initialized to {temperature_init_value}, "
              f"learnable={temperature_learnable}")
        
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Apply temperature scaling to attention computation.
        """
        # Compute attention scores: Q @ K^T
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # Apply scaling factor (original GPT-2 scaling)
        attn_weights = attn_weights / math.sqrt(value.size(-1))
        
        # Apply learnable/fixed temperature scaling per head
        # temperature_logits shape: [num_heads] 
        # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
        temperatures = torch.exp(self.temperature_logits)  # Convert from log space
        
        # Reshape for broadcasting: [1, num_heads, 1, 1]
        temperatures = temperatures.view(1, self.num_heads, 1, 1)
        
        # Apply temperature scaling
        attn_weights = attn_weights * temperatures
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights

    def get_temperature_info(self):
        """Get current temperature values for analysis."""
        with torch.no_grad():
            temperatures = torch.exp(self.temperature_logits).float()
            return {
                'layer_idx': self.layer_idx,
                'temperatures': temperatures.cpu().numpy().tolist(),
                'mean_temp': temperatures.mean().item(),
                'std_temp': temperatures.std().item(),
                'min_temp': temperatures.min().item(),
                'max_temp': temperatures.max().item(),
                'learnable': self.temperature_learnable,
                'init_value': self.temperature_init_value
            }

class TemperatureScaledGPT2Model(GPT2Model):
    """
    GPT-2 model with temperature-scaled attention layers.
    """
    
    def __init__(self, config, temperature_init_value=1.0, temperature_learnable=True):
        super().__init__(config)
        
        self.temperature_init_value = temperature_init_value
        self.temperature_learnable = temperature_learnable
        
        # Replace attention layers with temperature-scaled versions
        for i, layer in enumerate(self.h):
            layer.attn = TemperatureScaledGPT2Attention(
                config, 
                is_cross_attention=False, 
                layer_idx=i,
                temperature_init_value=temperature_init_value,
                temperature_learnable=temperature_learnable
            )
    
    def get_all_temperature_info(self):
        """Get temperature information from all layers."""
        temp_info = []
        for i, layer in enumerate(self.h):
            temp_info.append(layer.attn.get_temperature_info())
        return temp_info
    
    def set_temperature_learning(self, learnable):
        """Enable/disable temperature learning for all layers."""
        for layer in self.h:
            if hasattr(layer.attn, 'temperature_logits'):
                if learnable and not layer.attn.temperature_learnable:
                    # Convert buffer to parameter
                    temp_data = layer.attn.temperature_logits.data.clone()
                    delattr(layer.attn, 'temperature_logits')
                    layer.attn.temperature_logits = nn.Parameter(temp_data)
                    layer.attn.temperature_learnable = True
                elif not learnable and layer.attn.temperature_learnable:
                    # Convert parameter to buffer
                    temp_data = layer.attn.temperature_logits.data.clone()
                    delattr(layer.attn, 'temperature_logits')
                    layer.attn.register_buffer('temperature_logits', temp_data)
                    layer.attn.temperature_learnable = False

class TemperatureScaledGPT2LMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 Language Model with temperature-scaled attention.
    """
    
    def __init__(self, config, temperature_init_value=1.0, temperature_learnable=True):
        # We need to initialize the parent class first
        super().__init__(config)
        
        # Then replace the transformer with our temperature-scaled version
        self.transformer = TemperatureScaledGPT2Model(
            config, 
            temperature_init_value=temperature_init_value,
            temperature_learnable=temperature_learnable
        )
        
        # Initialize weights
        self.post_init()
    
    def get_all_temperature_info(self):
        """Get temperature information from all layers."""
        return self.transformer.get_all_temperature_info()
    
    def set_temperature_learning(self, learnable):
        """Enable/disable temperature learning for all layers."""
        self.transformer.set_temperature_learning(learnable)
    
    def get_temperature_parameters(self):
        """Get all temperature parameters for separate optimization if needed."""
        temp_params = []
        for layer in self.transformer.h:
            if (hasattr(layer.attn, 'temperature_logits') and 
                layer.attn.temperature_learnable and
                isinstance(layer.attn.temperature_logits, nn.Parameter)):
                temp_params.append(layer.attn.temperature_logits)
        return temp_params
    
    def get_non_temperature_parameters(self):
        """Get all non-temperature parameters for separate optimization if needed."""
        temp_param_ids = set(id(p) for p in self.get_temperature_parameters())
        non_temp_params = []
        for param in self.parameters():
            if id(param) not in temp_param_ids:
                non_temp_params.append(param)
        return non_temp_params

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained model with temperature scaling options.
        
        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            temperature_init_value: Initial temperature value (default: 1.0)
            temperature_learnable: Whether temperatures are learnable (default: True)
        """
        # Extract temperature-specific arguments
        temperature_init_value = kwargs.pop('temperature_init_value', 1.0)
        temperature_learnable = kwargs.pop('temperature_learnable', True)
        
        # Load the base model first
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Replace with temperature-scaled version
        config = model.config
        temp_model = cls(
            config,
            temperature_init_value=temperature_init_value,
            temperature_learnable=temperature_learnable
        )
        
        # Copy non-temperature parameters from the loaded model
        base_state_dict = model.state_dict()
        temp_state_dict = temp_model.state_dict()
        
        # Copy parameters that exist in both models (excluding temperature parameters)
        for key in base_state_dict:
            if key in temp_state_dict and 'temperature_logits' not in key:
                temp_state_dict[key] = base_state_dict[key]
        
        temp_model.load_state_dict(temp_state_dict, strict=False)
        
        return temp_model 