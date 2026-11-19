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
# ETHE v0.2 CONFIG
# =========================

SEED = 42

BATCH_SIZE = 16
BLOCK_SIZE = 128

EPOCHS = 30
LEARNING_RATE = 3e-4

VALIDATION_RATIO = 0.1

CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "ethe_v0_2_latest.pt"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =========================
# SEED
# =========================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 40)
print("ETHE v0.2 TRAINING")
print("=" * 40)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# LOAD TEXT
# =========================

with open(
    "data/dataset.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print(
    "Dataset characters:",
    len(text)
)


# =========================
# TOKENIZER
# =========================

tokenizer = CharacterTokenizer(
    text=text
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)

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
    int(
        len(dataset)
        * VALIDATION_RATIO
    )
)

training_size = (
    len(dataset)
    - validation_size
)


train_dataset, validation_dataset = (
    random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=torch.Generator()
        .manual_seed(SEED)
    )
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


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)


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


model = AetherModel(
    config
)

model.to(device)


# =========================
# PARAMETER COUNT
# =========================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()


# =========================
# RESUME
# =========================

start_epoch = 0

train_losses = []
validation_losses = []


if os.path.exists(
    CHECKPOINT_PATH
):

    print()
    print(
        "Checkpoint ditemukan."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    train_losses = checkpoint.get(
        "train_losses",
        []
    )

    validation_losses = checkpoint.get(
        "validation_losses",
        []
    )

    print(
        "Resume dari epoch:",
        start_epoch + 1
    )

else:

    print()
    print(
        "Tidak ada checkpoint."
    )

    print(
        "Training dimulai dari awal."
    )


# =========================
# TRAINING LOOP
# =========================

for epoch in range(
    start_epoch,
    EPOCHS
):

    model.train()

    total_train_loss = 0.0


    # =====================
    # TRAIN
    # =====================

    for inputs, targets in train_loader:

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            inputs
        )


        loss = criterion(
            logits.reshape(
                -1,
                config.vocab_size
            ),
            targets.reshape(
                -1
            )
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_train_loss += (
            loss.item()
        )


    average_train_loss = (
        total_train_loss
        / max(
            1,
            len(train_loader)
        )
    )


    # =====================
    # VALIDATION
    # =====================

    model.eval()

    total_validation_loss = 0.0


    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )


            logits = model(
                inputs
            )


            loss = criterion(
                logits.reshape(
                    -1,
                    config.vocab_size
                ),
                targets.reshape(
                    -1
                )
            )


            total_validation_loss += (
                loss.item()
            )


    average_validation_loss = (
        total_validation_loss
        / max(
            1,
            len(validation_loader)
        )
    )


    train_losses.append(
        average_train_loss
    )

    validation_losses.append(
        average_validation_loss
    )


    # =====================
    # PERPLEXITY
    # =====================

    try:

        perplexity = math.exp(
            min(
                average_validation_loss,
                20
            )
        )

    except OverflowError:

        perplexity = float("inf")


    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS} | "
        f"Train Loss: "
        f"{average_train_loss:.4f} | "
        f"Val Loss: "
        f"{average_validation_loss:.4f} | "
        f"Perplexity: "
        f"{perplexity:.2f}"
    )


    # =====================
    # SAVE CHECKPOINT
    # =====================

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_losses":
                train_losses,

            "validation_losses":
                validation_losses,

            "config": {
                "vocab_size":
                    config.vocab_size,

                "embedding_dim":
                    config.embedding_dim,

                "num_layers":
                    config.num_layers,

                "num_heads":
                    config.num_heads,

                "context_length":
                    config.context_length,

                "dropout":
                    config.dropout
            }
        },
        CHECKPOINT_PATH
    )

    print(
        "Checkpoint saved."
    )


print()
print("=" * 40)
print("ETHE v0.2 TRAINING SELESAI")
print("=" * 40)
print(
    "Checkpoint:",
    CHECKPOINT_PATH
)
