from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
import ipdb

from run_generation import set_seed
from run_generation import top_k_top_p_filtering
import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model.forward(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    hidden_states = outputs[-1][-1][:, -1, :].squeeze()
    return generated, hidden_states

class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, x):
        return F.sigmoid(self.linear(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default='<|endoftext|>',
                        help="Token at which text generation is stopped")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Setting config to output hidden states
    config_class = GPT2Config
    config = config_class.from_pretrained("gpt2")
    config.output_hidden_states = True

    tokenizer = tokenizer_class.from_pretrained(args.model_type)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    model.train()
    
    di = Discriminator().to(args.device)
    criterion = torch.nn.BCELoss()
    d_optim = torch.optim.Adam(di.parameters(), 1e-3, betas=(0.5, 0.999))
    di.load_state_dict(torch.load('datasets/discriminator.pt'))
    
    logger.info(args)
    train_df = pd.read_csv("datasets/train-samples.csv")

    # for i, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    diff, loss = [], []
    for j in range(10):
        for i, row in (train_df.iterrows()):
            d_optim.zero_grad()
            context_tokens = tokenizer.encode(row["text"][:2000], add_special_tokens=False)
            out, hidden_states = sample_sequence(
                model=model,
                context=context_tokens,
                num_samples=args.num_samples,
                length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                is_xlnet=bool(args.model_type == "xlnet"),
                is_xlm_mlm=False,
                xlm_mask_token=None,
                xlm_lang=None,
                device=args.device,
            )
            real = di.forward(hidden_states)
            real_loss = criterion(real, torch.tensor(1.0))
            real_loss.backward()

            context_tokens = tokenizer.encode(row["sample-0"][:2000], add_special_tokens=False)
            out, hidden_states = sample_sequence(
                model=model,
                context=context_tokens,
                num_samples=args.num_samples,
                length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                is_xlnet=bool(args.model_type == "xlnet"),
                is_xlm_mlm=False,
                xlm_mask_token=None,
                xlm_lang=None,
                device=args.device,
            )
            fake = di.forward(hidden_states)
            fake_loss = criterion(fake, torch.tensor(0.0))
            fake_loss.backward()
            d_optim.step()
            diff.append(float(real-fake))
            loss.append(float(real_loss+fake_loss))
            if i % 100 == 99:
                print("Step " + str(i) + ". Diff classified as: " + "{0:.3f}".format(np.mean(diff))
                      + ". Loss: " + "{0:.3f}".format(np.mean(loss)))
                torch.save(di.state_dict(), 'datasets/discriminator.pt')
                diff, loss = [], []

if __name__ == '__main__':
    main()