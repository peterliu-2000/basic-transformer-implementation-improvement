import os
import torch
from src.transformer_upgraded import TransformerUpgraded
from tokenizers import Tokenizer
from data.contexts import CONTEXTS
from src.utils import clean_bpe_text

## NOT SURE WHY FOR LOOP IS SO SLOW????

device = 'mps'
os.makedirs("data/generated_text_vanilla", exist_ok=True)

with open('data/finetuning_text.txt', 'r') as f:
    data = f.read()
tokenizer = Tokenizer.from_file('models/bpe_tokenizer.json')

model = TransformerUpgraded(vocab_size=tokenizer.get_vocab_size(), context_window_size=256).to(device)
model.load_state_dict(torch.load('models/transformer_finetuned.pth', map_location=device))
generated_text_dir = "data/generated_text_upgraded"
os.makedirs(generated_text_dir, exist_ok=True)

context = CONTEXTS[0]
context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
encoded_ids = tokenizer.encode(context).ids
decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
with torch.no_grad():
    cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output = clean_bpe_text(output)
output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{1}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[1]
context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
encoded_ids = tokenizer.encode(context).ids
decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
with torch.no_grad():
    cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output = clean_bpe_text(output)
output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{2}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[2]
context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
encoded_ids = tokenizer.encode(context).ids
decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
with torch.no_grad():
    cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output = clean_bpe_text(output)
output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{3}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[3]
context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
encoded_ids = tokenizer.encode(context).ids
decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
with torch.no_grad():
    cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output = clean_bpe_text(output)
output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{4}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[4]
context_tokens = torch.tensor(tokenizer.encode(context).ids, device=device).reshape(1, -1)
encoded_ids = tokenizer.encode(context).ids
decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
with torch.no_grad():
    cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output = clean_bpe_text(output)
output_file_path = os.path.join(generated_text_dir, f'bpe_token_upgraded_transformer_text_{5}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')




