import torch
from torch.utils.data import DataLoader
import argparse
import logging
from model import ImplicitModel
from data import CoTDataset, CoTDataCollator, extract_answer
import matplotlib.pyplot as plt
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def split_samples(input_ids, eos_token_id):
    # We assume that the input_ids share the same structure for each sample in the batch
    sep_positions = input_ids[0] == eos_token_id
    sep_positions = sep_positions.nonzero()
    diffs = [sep_positions[0] + 1] + list(sep_positions[1:] - sep_positions[:-1])
    diffs = [x.item() for x in diffs]
    inputs, cots, outputs = torch.split(input_ids, diffs, dim=1)
    return inputs, cots, outputs
    
    

@torch.no_grad()
def evaluate(dataloader, tokenizer, device, model, max_new_tokens):
    model.eval()
    total_samples = 0
    correct_at_id = {}
    for batch in dataloader:
        input_ids_all = batch['input_ids_all']
        # Remove answer part
        input_ids, cot_ids, targets_ids = split_samples(input_ids_all, tokenizer.eos_token_id)

        input_ids = input_ids.to(device)
        targets_ids = targets_ids.to(device)
    
        # Generate
        beam_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_on_two_eos=True,
        )

        inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        targets = tokenizer.batch_decode(targets_ids, skip_special_tokens=True)
        predicted_answers = tokenizer.batch_decode(beam_output, skip_special_tokens=True)

        def trim_answer(answer, split_pattern="####"):
            """
            We assume the answer is in the format of "#### <answer> #### or #### <answer>"
            """
            try:
                _, ans, _ = answer.strip().split(split_pattern, 2)
            except ValueError:
                _, ans = answer.strip().split(split_pattern, 1)
            ans = ans.replace(" ", "")
            return ans

        for inp, ans, target in zip(inputs, predicted_answers, targets):
            print(ans)
            trimmed_answer = trim_answer(ans)
            trimmed_target = trim_answer(target)
            print(trimmed_answer)
            print(trimmed_target)
            print("-"*100)
            for e, (a, t) in enumerate(zip(trimmed_answer, trimmed_target)):
                correct_at_id[e] = correct_at_id.get(e, 0) + (a == t)



        # # if max_new_tokens == targets_ids.shape[1]: # case for no CoT models
        # #     trimmed_answers = beam_output[:, input_ids.shape[1] + 3 : -1] # remove the input and initial <eos> tokens 
        # # else:
        # #     penultimate_eos_token = (beam_output[0] == tokenizer.eos_token_id).nonzero()[-2]
        # #     trimmed_answers = beam_output[:, penultimate_eos_token:]

        # trimmed_answers = beam_output[:, input_ids.shape[1] + 2 : -2] # remove the input and initial <eos> tokens 
        # trimmed_targets = targets_ids[:, 2:-2] # remove the initial special tokens and the final <eos> token


        # correct_per_index = torch.sum(trimmed_targets == trimmed_answers, dim=0)
        # for index, correct in enumerate(correct_per_index):
            # correct_at_id[index] = correct_at_id.get(index, 0) + correct.item()
        total_samples += input_ids.shape[0]

    return correct_at_id, total_samples


def plot_accuracy(correct_at_id, total_samples, rpath):
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    import numpy as np
    
    # Define matching colors for the plots
    blue_color = '#2E86AB'  # Deep blue
    red_color = '#A23B72'   # Deep red
    
    axs.bar(np.array(list(correct_at_id.keys()))-0.15, [x/total_samples for x in list(correct_at_id.values())], color=blue_color, width=0.3, alpha=0.5)
    ax = axs.twinx()
    ax.bar(np.array(list(correct_at_id.keys()))+0.15, [2, 5, 8, 11, 14, 13, 10, 7, 4, 1][:len(correct_at_id)], color=red_color, width=0.3, alpha=0.5)
    
    # Apply colors to axs elements
    axs.set_ylabel("Accuracy", color=blue_color)
    axs.set_ylim(0, 1)
    axs.tick_params(axis='y', labelcolor=blue_color)
    axs.yaxis.label.set_color(blue_color)
    
    # Apply colors to ax elements
    ax.set_ylabel("Number of operations per position", color=red_color)
    ax.tick_params(axis='y', labelcolor=red_color)
    ax.set_ylim(0, 15)

    axs.set_xticks(np.array(list(correct_at_id.keys())))
    axs.set_xticklabels([f'{x}' for x in np.array(list(correct_at_id.keys()))])
    axs.set_xlabel("Output token position")
    axs.set_title("Accuracy vs number of operations per position")
    plt.tight_layout()
    plt.savefig(os.path.join(rpath, "correct_at_id.png"), dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rpath', type=str, default=None)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--bf16', action='store_true')  
    parser.add_argument('--keep_k_target', type=int, default=0)
    
    parser.set_defaults(bf16=False)
    args = parser.parse_args()

    print (args)

    if args.bf16:
        dtype = 'bfloat16'
    else:
        dtype = 'float32'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (dtype, device)

    # Load model
    print (f'Loading from {args.rpath}')
    model = ImplicitModel.from_pretrained(args.rpath)
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation, keep_k_target=args.keep_k_target)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    correct_at_id, total_samples = evaluate(test_dataloader, tokenizer, device, model, args.max_new_tokens)
    plot_accuracy(correct_at_id, total_samples, args.rpath)

if __name__ == "__main__":
    main()
