from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=-1, max_size=-1, remove_cot=False, random_cot=False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print (f'Creating features from dataset file at {file_path}')
        eos_tok = tokenizer.eos_token


        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip().split('||') for line in f.readlines() if (len(line.strip()) > 0 and not line.strip().isspace()
                                                                             and len(line.strip().split('||')) ==2 )]
        if max_size > 0:
            print (f'truncated to {max_size}')
            lines = lines[:max_size]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if random_cot:
            example_cots = []
            for src, tgt in zip(src_lines, tgt_lines):
                cot = extract_cot(tgt)
                example_cots.append(cot)

        # Create a list of all examples
        self.examples_all = []
        for src, tgt in zip(src_lines, tgt_lines):
            ans = extract_answer(tgt)
            if remove_cot:
                sent = ' {} {} '.format(src, eos_tok) + ans + ' {}'.format(eos_tok)
            elif random_cot:
                import random
                cot = example_cots.pop(random.randint(0, len(example_cots) - 1))
                sent = ' {} {} '.format(src, eos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
            else:
                cot = extract_cot(tgt)
                sent = ' {} {} '.format(src, eos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)

            if max_length > 0:
                batch_encoding_all = tokenizer([sent], add_special_tokens=True, truncation=True, max_length=max_length)
            else:
                batch_encoding_all = tokenizer([sent], add_special_tokens=True)
            self.examples_all.append(batch_encoding_all["input_ids"][0])
        separator = tokenizer.eos_token_id
        self.separator = separator

    def __len__(self):
        return len(self.examples_all)

    def __getitem__(self, i):
        input_ids = self.examples_all[i]
        labels = copy.deepcopy(input_ids)
        sep_idx = labels.index(self.separator) + 1
        labels[:sep_idx] = [-100] * sep_idx
        return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                )
@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_all, labels_all = zip(*examples)
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        labels_all = self._tensorize_batch(labels_all)
        return {'input_ids_all': input_ids_all, 'labels_all': labels_all}

    def _tensorize_batch(self, examples):
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)
