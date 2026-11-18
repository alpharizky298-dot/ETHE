import json


class CharacterTokenizer:
    """
    Tokenizer karakter sederhana.
    Setiap karakter diberikan ID berupa angka.
    """

    def __init__(self, text=None, vocab=None):
        if vocab is not None:
            self.stoi = vocab
            self.itos = {int(v): k for k, v in vocab.items()}
        elif text is not None:
            chars = sorted(set(text))

            self.stoi = {
                char: index
                for index, char in enumerate(chars)
            }

            self.itos = {
                index: char
                for char, index in self.stoi.items()
            }
        else:
            raise ValueError("Berikan text atau vocab.")

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        """
        Teks -> list angka
        """
        unknown = [
            char for char in text
            if char not in self.stoi
        ]

        if unknown:
            raise ValueError(
                f"Karakter tidak ditemukan dalam vocabulary: "
                f"{sorted(set(unknown))}"
            )

        return [self.stoi[char] for char in text]

    def decode(self, tokens):
        """
        list angka -> teks
        """
        return "".join(self.itos[int(token)] for token in tokens)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                self.stoi,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as file:
            vocab = json.load(file)

        vocab = {
            char: int(index)
            for char, index in vocab.items()
        }

        return cls(vocab=vocab)
