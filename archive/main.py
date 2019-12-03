from datasets.encoder import get_codec
from model import GPT2, load_weight
from utils.utils import *
from utils.sample import *
import torch
import ipdb
import numpy as np
from transformers import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

titles = ["Muslims BUSTED: They Stole Millions In Gov’t Benefits",
# "30 Civilians Die In US Airstrike Called ‘To Protect US and Afghan troops’",
# "Clinton's Blitzkrieg Campaign: the Savage Politics of the Oligarchs",
# "Watch: Rigged Voting Machine Will Not Allow Vote For Trump/Pence… “Stuck” On Clinton/Kaine",
# "Stopping Hillary’s Coming War on Syria",
# "FBI Believe Clinton Foundation Case Moving Towards ‘Likely an Indictment’",
# "American Nightmare: Tyranny Under the FBI, an American Reality (Part III)",
"Girl Soldiers: Forgotten Casualties of War"]

def setup():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)

    codec = get_codec()
    model = GPT2(config)
    model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    if not os.path.exists('submit'):
        os.makedirs('submit')

    return codec, model, config

def main():
    codec, model, config = setup()
    from utils.sample import sample
    with open(os.path.join('submit', 'samples.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(titles)):
            start_text = titles[i]
            start_text = codec.encode(start_text).to(device)
            text = sample(model, start_text, config, codec)
            text = codec.decode(text.tolist()[0])

            f.write('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
            print('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            print("Prompt: " + titles[i])
            print(text)
    print("# Samples written to samples.txt.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
