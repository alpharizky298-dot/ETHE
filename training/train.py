import os
import random

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tokenizer.tokenizer import CharacterTokenizer
from training.dataset import TextDataset
from model.config import AetherConfig
from model.aether import AetherModel


# =========================
# CONFIGURATION
# =========================

SEED = 42

BATCH_SIZE = 16
BLOCK_SIZE = 64

EPOCHS = 20
LEARNING_RATE = 3e-4

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "aether_v0_1.pt"
)

TOKENIZER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocab.json"
)


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =========================
# LOAD DATA
# =========================

with open(
    "data/dataset.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()

print("Dataset characters:", len(text))


# =========================
# TOKENIZER
# =========================

tokenizer = CharacterTokenizer(text)

print("Vocabulary size:", tokenizer.vocab_size)

tokenizer.save(TOKENIZER_PATH)


# =========================
# DATASET
# =========================

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print("Training samples:", len(dataset))


# =========================
# MODEL
# =========================

config = AetherConfig(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    num_layers=2,
    num_heads=4,
    context_length=BLOCK_SIZE,
    dropout=0.1
)

model = AetherModel(config).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Model parameters:",
    f"{parameter_count:,}"
)


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# =========================
# TRAINING
# =========================

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )


# =========================
# SAVE CHECKPOINT
# =========================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "embedding_dim": config.embedding_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "context_length": config.context_length,
            "dropout": config.dropout
        }
    },
    CHECKPOINT_PATH
)

print()
print("================================")
print("Training selesai!")
print("Checkpoint:", CHECKPOINT_PATH)
print("Tokenizer :", TOKENIZER_PATH)
print("================================")
