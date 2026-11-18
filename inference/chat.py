import os

import torch

from tokenizer.tokenizer import CharacterTokenizer
from model.config import AetherConfig
from model.aether import AetherModel


CHECKPOINT_PATH = (
    "checkpoints/aether_v0_1.pt"
)

TOKENIZER_PATH = (
    "checkpoints/vocab.json"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# CHECK FILE
# =========================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        "Checkpoint belum ditemukan. "
        "Jalankan training terlebih dahulu."
    )

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        "Tokenizer belum ditemukan. "
        "Jalankan training terlebih dahulu."
    )


# =========================
# LOAD TOKENIZER
# =========================

tokenizer = CharacterTokenizer.load(
    TOKENIZER_PATH
)


# =========================
# LOAD CHECKPOINT
# =========================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

saved_config = checkpoint["config"]

config = AetherConfig(
    vocab_size=saved_config["vocab_size"],
    embedding_dim=saved_config["embedding_dim"],
    num_layers=saved_config["num_layers"],
    num_heads=saved_config["num_heads"],
    context_length=saved_config["context_length"],
    dropout=saved_config["dropout"]
)


# =========================
# LOAD MODEL
# =========================

model = AetherModel(config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print()
print("==============================")
print("       AETHER-1 v0.1")
print("==============================")
print("Device:", device)
print("Ketik 'exit' untuk keluar.")
print()


# =========================
# CHAT LOOP
# =========================

while True:

    prompt = input("You: ")

    if prompt.lower().strip() == "exit":
        print("AETHER: Sampai jumpa.")
        break

    if not prompt:
        continue

    try:
        tokens = tokenizer.encode(prompt)

    except ValueError as error:
        print("AETHER:", error)
        continue

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8
    )

    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print("AETHER:", generated_text)
    print()
