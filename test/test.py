import os
import sys
import tempfile

import torch

# Pastikan root project masuk Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from tokenizer.tokenizer import CharacterTokenizer
from training.dataset import TextDataset
from model.config import AetherConfig
from model.aether import AetherModel


print("=" * 40)
print("ETHE TEST SUITE")
print("=" * 40)


# =====================================
# TEST 1 — TOKENIZER
# =====================================

print("\n[1/5] Testing tokenizer...")

text = "Hello ETHE"

tokenizer = CharacterTokenizer(
    text=text
)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

assert decoded == text

print("Tokenizer: OK")


# =====================================
# TEST 2 — TOKENIZER SAVE/LOAD
# =====================================

print("\n[2/5] Testing tokenizer save/load...")

with tempfile.NamedTemporaryFile(
    suffix=".json",
    delete=False
) as temp_file:

    temp_path = temp_file.name


try:

    tokenizer.save(temp_path)

    loaded_tokenizer = (
        CharacterTokenizer.load(temp_path)
    )

    assert (
        loaded_tokenizer.encode(text)
        == encoded
    )

    assert (
        loaded_tokenizer.decode(encoded)
        == text
    )

    print("Tokenizer save/load: OK")

finally:

    if os.path.exists(temp_path):
        os.remove(temp_path)


# =====================================
# TEST 3 — DATASET
# =====================================

print("\n[3/5] Testing dataset...")

dataset_text = (
    "ETHE adalah model bahasa. "
    "ETHE belajar dari data. "
    "Model bahasa memprediksi token."
)

dataset_tokenizer = CharacterTokenizer(
    text=dataset_text
)

dataset = TextDataset(
    text=dataset_text,
    tokenizer=dataset_tokenizer,
    block_size=16
)

x, y = dataset[0]

assert x.shape == (16,)
assert y.shape == (16,)

assert x.dtype == torch.long
assert y.dtype == torch.long

print("Dataset: OK")
print("Input shape:", tuple(x.shape))
print("Target shape:", tuple(y.shape))


# =====================================
# TEST 4 — MODEL
# =====================================

print("\n[4/5] Testing model...")

config = AetherConfig(
    vocab_size=dataset_tokenizer.vocab_size,
    embedding_dim=128,
    num_layers=2,
    num_heads=4,
    context_length=16,
    dropout=0.1
)

model = AetherModel(config)

model.eval()

test_input = torch.randint(
    0,
    config.vocab_size,
    (2, 16)
)

with torch.no_grad():

    logits = model(test_input)


assert logits.shape == (
    2,
    16,
    config.vocab_size
)

print("Model forward: OK")
print("Output shape:", tuple(logits.shape))


# =====================================
# TEST 5 — GENERATION
# =====================================

print("\n[5/5] Testing generation...")

start_tokens = torch.randint(
    0,
    config.vocab_size,
    (1, 5)
)

with torch.no_grad():

    generated = model.generate(
        start_tokens,
        max_new_tokens=10,
        temperature=0.8
    )


assert generated.shape[0] == 1

assert generated.shape[1] == 15

print("Generation: OK")
print(
    "Generated shape:",
    tuple(generated.shape)
)


# =====================================
# RESULT
# =====================================

print()
print("=" * 40)
print("ALL TESTS PASSED")
print("=" * 40)
print("ETHE v0.2 basic components bekerja.")