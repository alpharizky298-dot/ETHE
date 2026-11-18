class Tokenizer:
    def __init__(self, text):
        # Ambil semua karakter unik
        chars = sorted(list(set(text)))

        # Karakter → angka
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Angka → karakter
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Mengubah teks menjadi angka."""
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        """Mengubah angka kembali menjadi teks."""
        return ''.join(self.itos[i] for i in tokens)

    @property
    def vocab_size(self):
        return len(self.stoi)
