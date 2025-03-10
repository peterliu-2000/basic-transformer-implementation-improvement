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

device = 'mps'

with open('data/training_text.txt', 'r', encoding='utf-8') as f:
    training_corpus = f.read()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    vocab_size = MAX_VOCAB_SIZE,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.pre_tokenizer = ByteLevel()

tokenizer.train(files = ['data/training_text.txt'], trainer = trainer)
training_encoded = tokenizer.encode(training_corpus).ids
vocab_size = tokenizer.get_vocab_size()
tokenizer.save('models/bpe_tokenizer.json')

print("Trained BPE Vocabulary Size:", vocab_size)

trans = TransformerUpgraded(
    vocab_size,
    context_window_size=CONTEXT_WINDOW_SIZE
).to(device)
trans.train()

learning_rate = 1e-4
optimizer = torch.optim.AdamW(trans.parameters(), lr=learning_rate)
eval_interval = 200
pretraining_loss_list = []

for it in tqdm(range(PRETRAIN_ITER)):
    xb, yb = get_batch_tokens(training_encoded, CONTEXT_WINDOW_SIZE, device)

    # evaluate the loss
    logits, loss = trans(xb, yb)
    pretraining_loss_list.append(loss.detach().item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(trans.state_dict(), 'models/transformer_pretrained.pth')

# Plot the loss values
plt.plot(pretraining_loss_list, label="Training Loss")

# Add labels and title
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Upgraded Transformer Pretraining Loss")
plt.legend()
plt.savefig('plots/transformer_pretraining_loss.png')