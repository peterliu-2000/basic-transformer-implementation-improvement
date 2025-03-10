import torch
from tqdm import tqdm
from src.transformer_upgraded import TransformerUpgraded
from src.utils import get_batch_tokens
from src.GLOBAL_VARS import CONTEXT_WINDOW_SIZE, MAX_VOCAB_SIZE, FINETUNE_ITER
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

# Device setup
device = 'mps'

tokenizer = Tokenizer.from_file('models/bpe_tokenizer.json')
assert tokenizer.get_vocab_size() == MAX_VOCAB_SIZE

model = TransformerUpgraded(MAX_VOCAB_SIZE, context_window_size=CONTEXT_WINDOW_SIZE).to(device)
model.load_state_dict(torch.load('models/transformer_pretrained.pth', map_location=device))
model.train()  # Set to training mode

# Load & Tokenize New Training Data
with open('data/finetuning_text.txt', 'r', encoding='utf-8') as f:
    fine_tune_corpus = f.read()

encoded = tokenizer.encode(fine_tune_corpus).ids
encoded = torch.tensor(encoded, dtype=torch.long).to(device)

# Optimizer & Hyperparameters
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
finetuning_loss_list = []

# Fine-Tuning Loop
for it in tqdm(range(FINETUNE_ITER)):
    xb, yb = get_batch_tokens(encoded, CONTEXT_WINDOW_SIZE, device)

    logits, loss = model(xb, yb)
    finetuning_loss_list.append(loss.detach().item())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the Fine-Tuned Model
torch.save(model.state_dict(), 'models/transformer_finetuned.pth')

# Plot the loss values
plt.plot(finetuning_loss_list, label="Finetuning Loss")

# Add labels and title
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Upgraded Transformer Finetuning Loss")
plt.legend()
plt.savefig("plots/transformer_finetuned_loss.png", dpi=300, bbox_inches="tight")

