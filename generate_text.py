import os
import torch
from tqdm import tqdm
from src.char_tokenizer import CharTokenizer
from src.transformer_vanilla import TransformerVanilla
from src.transformer_upgraded import TransformerUpgraded
from tokenizers import Tokenizer
from data.contexts import CONTEXTS
from src.utils import clean_bpe_text

device = 'mps'

def generate_vanilla_text():
    print("Generating text with vanilla transformer...")
    # Setup directories
    generated_text_dir = "data/generated_text_vanilla"
    os.makedirs(generated_text_dir, exist_ok=True)

    # Initialize character tokenizer
    with open('data/finetuning_text.txt', 'r') as f:
        data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    tokenizer = CharTokenizer(chars)

    # Load model
    model = TransformerVanilla(vocab_size=vocab_size, context_window_size=256).to(device)
    model.load_state_dict(torch.load('models/transformer_vanilla.pth', map_location=device))
    model.eval()

    # Generate text for each context
    for idx, context in enumerate(tqdm(CONTEXTS, desc="Generating vanilla texts")):
        context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
        cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
        cond_gen = cond_gen[0].tolist()
        output = tokenizer.decode(cond_gen)
        
        output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{idx + 1}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(output)

def generate_upgraded_text():
    print("Generating text with upgraded transformer...")
    # Setup directories
    generated_text_dir = "data/generated_text_upgraded"
    os.makedirs(generated_text_dir, exist_ok=True)

    # Load BPE tokenizer
    tokenizer = Tokenizer.from_file('models/bpe_tokenizer.json')

    # Load model
    model = TransformerUpgraded(vocab_size=tokenizer.get_vocab_size(), context_window_size=256).to(device)
    model.load_state_dict(torch.load('models/transformer_finetuned.pth', map_location=device))
    model.eval()

    # Generate text for each context
    for idx, context in enumerate(tqdm(CONTEXTS, desc="Generating upgraded texts")):
        context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
        
        with torch.no_grad():
            cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
        
        cond_gen = cond_gen[0].tolist()
        output = tokenizer.decode(cond_gen)
        output = clean_bpe_text(output)
        
        output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{idx + 1}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(output)

def main():
    # Create parent directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    try:
        generate_vanilla_text()
        print("Vanilla text generation completed!")
    except Exception as e:
        print(f"Error in vanilla text generation: {str(e)}")

    try:
        generate_upgraded_text()
        print("Upgraded text generation completed!")
    except Exception as e:
        print(f"Error in upgraded text generation: {str(e)}")

if __name__ == "__main__":
    main() 