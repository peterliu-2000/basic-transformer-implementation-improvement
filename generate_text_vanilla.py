import os
import torch
from src.char_tokenizer import CharTokenizer
from src.transformer_vanilla import TransformerVanilla
from data.contexts import CONTEXTS

## NOT SURE WHY FOR LOOP IS SO SLOW????

device = 'mps'
os.makedirs("data/generated_text_vanilla", exist_ok=True)

with open('data/finetuning_text.txt', 'r') as f:
    data = f.read()
chars = sorted(list(set(data)))
vocab_size = len(chars)
tokenizer = CharTokenizer(chars)

model = TransformerVanilla(vocab_size=vocab_size, context_window_size=256).to(device)
model.load_state_dict(torch.load('models/transformer_vanilla.pth', map_location=device))
generated_text_dir = "data/generated_text_vanilla"

context = CONTEXTS[0]
context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{1}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[1]
context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{2}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[2]
context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{3}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[3]
context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{4}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')

context = CONTEXTS[4]
context_tokens = torch.tensor(tokenizer.encode(context), device=device).reshape(1, -1)
cond_gen, perplexity = model.generate(context_tokens, max_new_tokens=256)
cond_gen = cond_gen[0].tolist()
output = tokenizer.decode(cond_gen)
output_file_path = os.path.join(generated_text_dir, f'char_token_vanilla_transformer_text_{5}.txt')
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(output)
print('done')
print("Text generation completed!")





