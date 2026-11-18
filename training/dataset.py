import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        text,
        tokenizer,
        block_size
    ):
        self.block_size = block_size

        tokens = tokenizer.encode(text)

        if len(tokens) <= block_size:
            raise ValueError(
                "Dataset terlalu pendek untuk block_size."
            )

        self.data = torch.tensor(
            tokens,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):

        x = self.data[
            index:index + self.block_size
        ]

        y = self.data[
            index + 1:index + self.block_size + 1
        ]

        return x, y
