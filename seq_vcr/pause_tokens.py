import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Union
from transformers import AutoTokenizer
from .config import SeqVCRConfig


class PauseTokenProcessor:
    """
    Handles pause token insertion into input sequences for Seq-VCR
    
    Pause tokens serve as substitutes for chain-of-thought (CoT) tokens,
    providing additional computation time without requiring explicit supervision.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, config: SeqVCRConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.num_pause_tokens = config.num_pause_tokens
        
        # Set up pause tokens
        self.pause_token = config.pause_token
        self.pause_start_token = config.pause_start_token
        self.pause_end_token = config.pause_end_token
        
        # Add tokens to tokenizer if needed and get their IDs
        self._setup_pause_tokens()
    
    def _setup_pause_tokens(self):
        """Add pause tokens to tokenizer vocabulary if not present"""
        tokens_to_add = []
        
        # Check if tokens already exist
        if self.pause_token not in self.tokenizer.get_vocab():
            tokens_to_add.append(self.pause_token)
        if self.pause_start_token not in self.tokenizer.get_vocab():
            tokens_to_add.append(self.pause_start_token)
        if self.pause_end_token not in self.tokenizer.get_vocab():
            tokens_to_add.append(self.pause_end_token)
        
        # Add tokens if needed
        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add)
            print(f"Added pause tokens to tokenizer: {tokens_to_add}")
        
        # Get token IDs
        self.pause_id = self.tokenizer.convert_tokens_to_ids(self.pause_token)
        self.pause_start_id = self.tokenizer.convert_tokens_to_ids(self.pause_start_token)
        self.pause_end_id = self.tokenizer.convert_tokens_to_ids(self.pause_end_token)
    
    def add_pause_tokens(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Insert pause tokens between input and CoT/output
        
        Format: input <eos> </pause_start> <pause> <pause> </pause_end> output <eos>
        
        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] - optional, will be modified to set pause token labels to -100
        
        Returns:
            enhanced_input_ids: [batch_size, enhanced_seq_len]
            enhanced_labels: [batch_size, enhanced_seq_len] if labels provided
        """
        if not self.config.enable_pause_tokens or self.num_pause_tokens == 0:
            # Return original sequences if pause tokens are disabled
            if labels is not None:
                return input_ids, labels
            else:
                return input_ids
        
        batch_size = input_ids.shape[0]
        
        # Find first EOS position (end of input)
        eos_positions = self._find_first_eos_positions(input_ids)
        
        # Create pause token sequence
        pause_sequence = [self.pause_start_id] + [self.pause_id] * self.num_pause_tokens + [self.pause_end_id]
        pause_tensor = torch.tensor(pause_sequence, dtype=input_ids.dtype, device=input_ids.device)
        
        # Insert pause tokens after first EOS
        enhanced_sequences = []
        enhanced_label_sequences = []
        
        for i in range(batch_size):
            eos_pos = eos_positions[i].item()
            original_seq = input_ids[i]
            
            # Create enhanced sequence
            enhanced_seq = torch.cat([
                original_seq[:eos_pos+1],           # input + first eos
                pause_tensor,                       # pause tokens
                original_seq[eos_pos+1:]           # rest of sequence
            ])
            enhanced_sequences.append(enhanced_seq)
            
            # Handle labels: set pause token positions to -100 (ignore in loss)
            if labels is not None:
                original_labels = labels[i]
                pause_labels = torch.full((len(pause_sequence),), -100, 
                                        dtype=original_labels.dtype, device=original_labels.device)
                enhanced_labels = torch.cat([
                    original_labels[:eos_pos+1],     # input labels
                    pause_labels,                    # pause tokens -> -100 (ignored)
                    original_labels[eos_pos+1:]     # output labels
                ])
                enhanced_label_sequences.append(enhanced_labels)
        
        # Pad sequences to same length
        enhanced_input_ids = pad_sequence(
            enhanced_sequences, 
            batch_first=True, 
            padding_value=self.tokenizer.eos_token_id
        )
        
        if labels is not None:
            enhanced_labels = pad_sequence(
                enhanced_label_sequences, 
                batch_first=True, 
                padding_value=-100
            )
            return enhanced_input_ids, enhanced_labels
        else:
            return enhanced_input_ids
    
    def _find_first_eos_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find the position of the first EOS token in each sequence"""
        eos_mask = (input_ids == self.tokenizer.eos_token_id)
        
        # Find first occurrence of EOS in each sequence
        eos_positions = []
        for i in range(input_ids.shape[0]):
            eos_indices = torch.where(eos_mask[i])[0]
            if len(eos_indices) > 0:
                eos_positions.append(eos_indices[0])
            else:
                # If no EOS found, put pause tokens at the end
                eos_positions.append(len(input_ids[i]) - 1)
        
        return torch.tensor(eos_positions, device=input_ids.device)
    
    def remove_pause_tokens(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Remove pause tokens from generated sequences for evaluation
        
        Args:
            sequences: [batch_size, seq_len] - sequences potentially containing pause tokens
            
        Returns:
            cleaned_sequences: [batch_size, cleaned_seq_len] - sequences without pause tokens
        """
        if not self.config.enable_pause_tokens:
            return sequences
        
        # Create mask for pause tokens
        pause_token_ids = {self.pause_id, self.pause_start_id, self.pause_end_id}
        
        cleaned_sequences = []
        for seq in sequences:
            # Remove pause tokens
            mask = ~torch.isin(seq, torch.tensor(list(pause_token_ids), device=seq.device))
            cleaned_seq = seq[mask]
            cleaned_sequences.append(cleaned_seq)
        
        # Pad to same length
        return pad_sequence(cleaned_sequences, batch_first=True, padding_value=self.tokenizer.eos_token_id)
    
    def get_pause_token_info(self) -> dict:
        """Get information about pause tokens for debugging"""
        return {
            'enabled': self.config.enable_pause_tokens,
            'num_pause_tokens': self.num_pause_tokens,
            'pause_token': self.pause_token,
            'pause_start_token': self.pause_start_token,
            'pause_end_token': self.pause_end_token,
            'pause_id': self.pause_id,
            'pause_start_id': self.pause_start_id,
            'pause_end_id': self.pause_end_id,
            'vocab_size': len(self.tokenizer)
        } 