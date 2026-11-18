import os
import tempfile

import torch

from tokenizer.tokenizer import CharacterTokenizer
from training.dataset import TextDataset
from model.config import AetherConfig
from model.aether import AetherModel


print("=== TEST 1: TOKENIZER ===")

text = "AETHER adalah AI."

tokenizer = CharacterTokenizer(text)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

assert decoded == text

print("Tokenizer:", "OK")


print()
print("=== TEST 2: DATASET ===")

dataset_text = (
    "AETHER adalah proyek AI. "
    "AETHER belajar dari data. "
    "AI menggunakan neural network."
)

tokenizer = CharacterTokenizer(dataset_text)

dataset = TextDataset(
    dataset_text,
    tokenizer,
    block_size=16
)

x, y = dataset[0]

assert x.shape == y.shape
assert x.shape[0] == 16

print("Dataset:", "OK")


print()
print("=== TEST 3: MODEL ===")

config = AetherConfig(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    num_layers=2,
    num_heads=4,
    context_length=16,
    dropout=0.1
)

model = AetherModel(config)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (2, 16)
)

logits = model(input_ids)

assert logits.shape == (
    2,
    16,
    config.vocab_size
)

print("Model:", "OK")


print()
print("=== TEST 4: GENERATION ===")

generated = model.generate(
    input_ids[:, :4],
    max_new_tokens=5
)

assert generated.shape[1] == 9

print("Generation:", "OK")


print()
print("=== TEST 5: SAVE TOKENIZER ===")

with tempfile.NamedTemporaryFile(
    suffix=".json",
    delete=False
) as file:

    temp_path = file.name

tokenizer.save(temp_path)

loaded = CharacterTokenizer.load(
    temp_path
)

assert loaded.decode(
    loaded.encode(dataset_text)
) == dataset_text

os.remove(temp_path)

print("Tokenizer save/load:", "OK")


print()
print("==============================")
print("ALL TESTS PASSED!")
print("==============================")
