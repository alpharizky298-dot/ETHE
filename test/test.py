import os
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tokenizer.tokenizer import CharacterTokenizer
from training.dataset import TextDataset
from model.config import AetherConfig
from model.aether import AetherModel


# =========================
# CONFIG
# =========================

SEED = 42

BATCH_SIZE = 16
BLOCK_SIZE = 64

EPOCHS = 20
LEARNING_RATE = 3e-4

VALIDATION_RATIO = 0.1

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "aether_v0_2_latest.pt"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# LOAD DATASET
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

tokenizer = CharacterTokenizer(text=text)

print("Vocabulary size:", tokenizer.vocab_size)

tokenizer.save(
    os.path.join(
        CHECKPOINT_DIR,
        "vocab.json"
    )
)


# =========================
# DATASET
# =========================

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE
)

validation_size = max(
    1,
    int(len(dataset) * VALIDATION_RATIO)
)

training_size = len(dataset) - validation_size

train_dataset, validation_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


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

model = AetherModel(config)
model.to(device)


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()


# =========================
# COUNT PARAMETERS
# =========================

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Parameters:", parameter_count)


# =========================
# RESUME CHECKPOINT
# =========================

start_epoch = 0
train_losses = []
validation_losses = []

if os.path.exists(CHECKPOINT_PATH):

    print()
    print("Checkpoint ditemukan.")
    print("Melanjutkan training...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    train_losses = checkpoint.get(
        "train_losses",
        []
    )

    validation_losses = checkpoint.get(
        "validation_losses",
        []
    )

    print(
        "Melanjutkan dari epoch:",
        start_epoch + 1
    )

else:

    print()
    print("Checkpoint tidak ditemukan.")
    print("Memulai training dari awal.")


# =========================
# TRAINING
# =========================

for epoch in range(start_epoch, EPOCHS):

   