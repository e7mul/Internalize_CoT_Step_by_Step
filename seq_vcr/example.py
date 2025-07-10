#!/usr/bin/env python3
"""
Example usage of Seq-VCR implementation

This script demonstrates how to use the Seq-VCR regularization with
the existing ImplicitModel without modifying the original codebase.
"""

import sys
import os
import torch

# Add paths to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from seq_vcr import SeqVCRImplicitModel, SeqVCRConfig
from src.model import ImplicitModel
from src.configuration_model import ImplicitModelConfig


def create_example_config():
    """Create example configuration for testing"""
    # Create base model config (modify these paths as needed)
    base_config = ImplicitModelConfig(
        base_model="gpt2",  # or path to your model
        tokenizer_name="gpt2"  # or path to your tokenizer
    )
    return base_config


def example_basic_usage():
    """Example 1: Basic Seq-VCR usage for multiplication task"""
    print("=== Example 1: Basic Seq-VCR for Multiplication ===")
    
    # Create base model
    base_config = create_example_config()
    base_model = ImplicitModel(base_config)
    
    # Create Seq-VCR configuration for multiplication task
    seq_vcr_config = SeqVCRConfig.for_multiplication_task(
        enable_seq_vcr=True,
        enable_pause_tokens=True,
        num_pause_tokens=2
    )
    
    # Wrap the base model with Seq-VCR capabilities
    model = SeqVCRImplicitModel(base_model, seq_vcr_config)
    
    print(f"Seq-VCR enabled: {seq_vcr_config.enable_seq_vcr}")
    print(f"Lambda var: {seq_vcr_config.lambda_var}")
    print(f"Lambda cov: {seq_vcr_config.lambda_cov}")
    print(f"Pause tokens enabled: {seq_vcr_config.enable_pause_tokens}")
    print(f"Number of pause tokens: {seq_vcr_config.num_pause_tokens}")
    
    # Example input (replace with actual data)
    input_text = "12 * 34"
    inputs = model.tokenizer(input_text, return_tensors="pt")
    
    print(f"Original vocab size: {base_model.base_model.config.vocab_size}")
    print(f"Current vocab size: {len(model.tokenizer)}")
    
    return model


def example_with_training_simulation():
    """Example 2: Simulate training with regularization loss"""
    print("\n=== Example 2: Training Simulation ===")
    
    # Create model with Seq-VCR
    base_config = create_example_config()
    base_model = ImplicitModel(base_config)
    
    seq_vcr_config = SeqVCRConfig.for_multiplication_task(
        enable_seq_vcr=True,
        enable_pause_tokens=True
    )
    
    model = SeqVCRImplicitModel(base_model, seq_vcr_config)
    model.train()  # Set to training mode
    
    # Simulate batch data
    batch_size = 4
    seq_length = 20
    vocab_size = len(model.tokenizer)
    
    # Create dummy input and labels
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
    labels = input_ids.clone()
    labels[:, :seq_length//2] = -100  # Ignore first half in loss
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Forward pass with regularization
    outputs = model.compute_loss(input_ids, labels)
    
    print(f"Total loss: {outputs.loss.item():.4f}")
    print(f"Regularization loss: {model.last_regularization_loss:.4f}")
    
    # Get detailed regularization info
    reg_info = model.get_regularization_info()
    print("Regularization info:")
    for key, value in reg_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return model


def example_pause_tokens():
    """Example 3: Demonstrate pause token processing"""
    print("\n=== Example 3: Pause Token Processing ===")
    
    base_config = create_example_config()
    base_model = ImplicitModel(base_config)
    
    seq_vcr_config = SeqVCRConfig(
        enable_seq_vcr=False,  # Focus on pause tokens only
        enable_pause_tokens=True,
        num_pause_tokens=3
    )
    
    model = SeqVCRImplicitModel(base_model, seq_vcr_config)
    
    # Get pause token info
    pause_info = model.get_pause_token_info()
    print("Pause token info:")
    for key, value in pause_info.items():
        print(f"  {key}: {value}")
    
    # Example with text
    input_text = "What is 12 * 34?"
    inputs = model.tokenizer(input_text + model.tokenizer.eos_token, return_tensors="pt")
    
    print(f"\nOriginal input: {input_text}")
    print(f"Original tokens: {model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    
    # Add pause tokens
    enhanced_ids = model.pause_processor.add_pause_tokens(inputs['input_ids'])
    enhanced_tokens = model.tokenizer.convert_ids_to_tokens(enhanced_ids[0])
    
    print(f"Enhanced tokens: {enhanced_tokens}")
    
    return model


def example_regularization_only():
    """Example 4: Use only regularization without pause tokens"""
    print("\n=== Example 4: Regularization Only ===")
    
    base_config = create_example_config()
    base_model = ImplicitModel(base_config)
    
    seq_vcr_config = SeqVCRConfig.for_arithmetic_task(
        enable_seq_vcr=True,
        enable_pause_tokens=False  # No pause tokens
    )
    
    model = SeqVCRImplicitModel(base_model, seq_vcr_config)
    
    print(f"Regularization enabled: {seq_vcr_config.enable_seq_vcr}")
    print(f"Pause tokens enabled: {seq_vcr_config.enable_pause_tokens}")
    print(f"Lambda var: {seq_vcr_config.lambda_var}")
    print(f"Lambda cov: {seq_vcr_config.lambda_cov}")
    
    return model


if __name__ == "__main__":
    print("Seq-VCR Implementation Examples")
    print("================================")
    
    try:
        # Run examples
        model1 = example_basic_usage()
        model2 = example_with_training_simulation()
        model3 = example_pause_tokens()
        model4 = example_regularization_only()
        
        print("\n=== All examples completed successfully! ===")
        print("\nTo use Seq-VCR in your training:")
        print("1. Create your base ImplicitModel as usual")
        print("2. Create a SeqVCRConfig with desired parameters")
        print("3. Wrap with SeqVCRImplicitModel")
        print("4. Use model.compute_loss() for training")
        print("5. Use model.generate() for inference")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have the required dependencies installed")
        print("and the paths to src/ directory are correct")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This is expected if you don't have a trained model or proper config") 