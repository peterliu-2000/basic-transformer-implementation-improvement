from src.transformer_upgraded import TransformerUpgraded
from transformers import AutoTokenizer
from tqdm import tqdm
from src.utils import *
from src.GLOBAL_VARS import *
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import re
import os
import torch

device = 'mps'

def train_tokenizer():
    print("Training BPE tokenizer...")
    with open('data/training_text.txt', 'r', encoding='utf-8') as f:
        training_corpus = f.read()

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=MAX_VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.pre_tokenizer = ByteLevel()

    tokenizer.train(files=['data/training_text.txt'], trainer=trainer)
    vocab_size = tokenizer.get_vocab_size()
    tokenizer.save('models/bpe_tokenizer.json')
    
    print(f"Trained BPE Vocabulary Size: {vocab_size}")
    return tokenizer, training_corpus, vocab_size

def pretrain_model(tokenizer, training_corpus, vocab_size):
    print("Starting pretraining...")
    training_encoded = tokenizer.encode(training_corpus).ids
    
    model = TransformerUpgraded(
        vocab_size,
        context_window_size=CONTEXT_WINDOW_SIZE
    ).to(device)
    model.train()

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pretraining_loss_list = []

    for it in tqdm(range(PRETRAIN_ITER), desc="Pretraining"):
        xb, yb = get_batch_tokens(training_encoded, CONTEXT_WINDOW_SIZE, device)
        logits, loss = model(xb, yb)
        pretraining_loss_list.append(loss.detach().item())
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'models/transformer_pretrained.pth')
    
    # Plot pretraining loss
    plt.figure(figsize=(10, 5))
    plt.plot(pretraining_loss_list, label="Pretraining Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Upgraded Transformer Pretraining Loss")
    plt.legend()
    plt.savefig('plots/transformer_pretraining_loss.png')
    plt.close()
    
    return model

def finetune_model(model, tokenizer):
    print("Starting finetuning...")
    model.train()

    # Load & Tokenize Finetuning Data
    with open('data/finetuning_text.txt', 'r', encoding='utf-8') as f:
        fine_tune_corpus = f.read()

    encoded = tokenizer.encode(fine_tune_corpus).ids
    encoded = torch.tensor(encoded, dtype=torch.long).to(device)

    # Optimizer & Training Setup
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    finetuning_loss_list = []

    # Fine-Tuning Loop
    for it in tqdm(range(FINETUNE_ITER), desc="Finetuning"):
        xb, yb = get_batch_tokens(encoded, CONTEXT_WINDOW_SIZE, device)
        logits, loss = model(xb, yb)
        finetuning_loss_list.append(loss.detach().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the Fine-Tuned Model
    torch.save(model.state_dict(), 'models/transformer_finetuned.pth')

    # Plot finetuning loss
    plt.figure(figsize=(10, 5))
    plt.plot(finetuning_loss_list, label="Finetuning Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Upgraded Transformer Finetuning Loss")
    plt.legend()
    plt.savefig("plots/transformer_finetuned_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Step 1: Train tokenizer
    tokenizer, training_corpus, vocab_size = train_tokenizer()
    
    # Step 2: Pretrain model
    model = pretrain_model(tokenizer, training_corpus, vocab_size)
    
    # Step 3: Finetune model
    finetune_model(model, tokenizer)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 