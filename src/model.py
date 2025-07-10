import os
import torch
import torch.nn as nn
import math
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList

from configuration_model import ImplicitModelConfig
from utils import DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor


class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False):
        super().__init__()
        self.config = config
        
        # Choose between standard and temperature-scaled model
        if config.use_temperature_scaling:
            from custom_gpt2 import TemperatureScaledGPT2LMHeadModel
            self.base_model = TemperatureScaledGPT2LMHeadModel.from_pretrained(
                config.base_model,
                temperature_init_value=config.temperature_init_value,
                temperature_learnable=config.temperature_learnable,
                trust_remote_code=True
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model, 
                trust_remote_code=True
            )
            
        if reinitialize_weights:
            print ('Reinitializing model weights!')
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def forward(self, input_ids, output_attentions=False, **kwargs):
        outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions, **kwargs)
        return outputs

    def compute_loss(self, input_ids, labels, output_attentions=False):
        outputs = self.forward(input_ids=input_ids, output_attentions=output_attentions)
        logits = outputs.logits

        def get_last_sep_position(input_ids):
            """
            Get the index of the second-to-last True value in each row.
            Returns -1 for rows with fewer than 2 True values.
            """
            # Get all True positions
            row_indices, col_indices = input_ids.eq(self.tokenizer.eos_token_id).nonzero(as_tuple=True)
            
            results = []
            # Group by row
            for row in range(input_ids.shape[0]):
                # Get all True positions for this row
                row_mask = (row_indices == row)
                if row_mask.sum() >= 2:
                    # Get column indices for this row and take second-to-last
                    row_cols = col_indices[row_mask]
                    results.append(row_cols[-2])
                else:
                    results.append(-1)
            return torch.tensor(results)

        sep_positions = get_last_sep_position(input_ids)
        assert len(sep_positions.unique()) == 1, 'sep_positions has more than one unique value'
        sep_position = sep_positions[0]
        ans_preds = logits[..., sep_position:-1, :].argmax(-1)
        ans_labels = labels[..., sep_position+1:]
        correct_ans_tokens = (ans_preds == ans_labels).sum()
        total_ans_tokens = (ans_labels != -100).sum()
        correct_per_row = (ans_preds == ans_labels).sum(-1)
        total_correct_answers = (correct_per_row == ans_labels.shape[-1]).sum()
    
        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        outputs.total_correct_answers = total_correct_answers
        outputs.correct_ans_tokens = correct_ans_tokens
        outputs.total_ans_tokens = total_ans_tokens
        return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True):
        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None
        

        beam_output = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=None,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            num_return_sequences=1,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            do_sample=True,
        )
        return beam_output

    def get_temperature_info(self):
        """Get temperature information if using temperature scaling."""
        if hasattr(self.base_model, 'get_all_temperature_info'):
            return self.base_model.get_all_temperature_info()
        else:
            return None

    def set_temperature_learning(self, learnable):
        """Enable/disable temperature learning."""
        if hasattr(self.base_model, 'set_temperature_learning'):
            self.base_model.set_temperature_learning(learnable)
        else:
            print("Temperature scaling not enabled for this model")

    @classmethod
    def from_pretrained(cls, pretrained_path, override_config=None):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        if override_config is not None:
            config.update(override_config)
        model = ImplicitModel(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if override_config is not None:
                print(f"Error loading state_dict: {e}")
                print(f"Overriding config: {override_config}")
                model.load_state_dict(state_dict, strict=False)
            else:
                raise e

        if override_config is not None and override_config['use_temperature_scaling']:
            for name, module in model.named_modules():
                if hasattr(module, 'temperature_logits'):
                    module.temperature_logits.data = torch.full((module.num_heads,), math.log(override_config['temperature_init_value']), dtype=torch.float32)

        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
