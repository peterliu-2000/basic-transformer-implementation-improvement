import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())


# Define file paths
vanilla_folder = 'data/generated_text_vanilla'
upgraded_folder = "data/generated_text_upgraded"
vanilla_files = [f"char_token_vanilla_transformer_text_{i}.txt" for i in range(1, 6)]
upgraded_files = [f"bpe_token_upgraded_transformer_text_{i}.txt" for i in range(1, 6)]

results = {}

for i in range(5):
    vanilla_path = os.path.join(vanilla_folder, vanilla_files[i])
    upgraded_path = os.path.join(upgraded_folder, upgraded_files[i])

    # Read texts
    with open(vanilla_path, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(upgraded_path, "r", encoding="utf-8") as f:
        text2 = f.read()

    # Compute perplexities
    perplexity_text1 = calculate_perplexity(text1)
    perplexity_text2 = calculate_perplexity(text2)

    results[f"pair_{i + 1}"] = {
        "vanilla_perplexity": perplexity_text1,
        "upgraded_perplexity": perplexity_text2
    }

    print(f"Pair {i + 1} - Vanilla: {perplexity_text1:.2f}, Upgraded: {perplexity_text2:.2f}")

# Save results to JSON file
output_path = "data/perplexity_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_path}")
