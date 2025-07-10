import torch
import argparse
import logging
from model import ImplicitModel
import os

from logit_lens_v2.logit_lens_utils import generate_fast, generate_token_progress_plot
from logit_lens_v2.logit_lens import Logit_Lens

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rpath', type=str, default=None)
    parser.add_argument('--bf16', action='store_true')  
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



    prompt = "3 5 4 1 1 * 2 1 4 6 5||"
    tokenizer.pad_token = tokenizer.eos_token
    # generate_fast(model, tokenizer, prompt, top_k = 10, max_out_len = 100, arg_max_greedy=True, debug=True)

    logit_lens = Logit_Lens(
        model.base_model, tokenizer,
    )
    generated_tokens, v_space_reprs, output_vectors, past_key_values = logit_lens.generate_next_token(
        prompt, 
        # debug = True,
        with_previous = True
    )
    generated_tokens, v_space_reprs, output_vectors, original_prompt_tokenized = logit_lens.generate_argmax_greedy(
        prompt,
        max_out_len=24
    )
    generation_start_position = len(tokenizer.encode(prompt))

    prompt + " --" + "".join([token[0] for token in generated_tokens[generation_start_position - 1: ]])
    plotly_fig = generate_token_progress_plot(
                prompt, original_prompt_tokenized,
                generated_tokens, v_space_reprs, 
                layer_skip = 0, start_idx = 0, end_idx=24
            )
    plotly_fig.write_html(os.path.join(args.rpath, "plot.html"))
    plotly_fig

if __name__ == "__main__":
    main()
