class AetherConfig:

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        context_length=128,
        dropout=0.1
    ):
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim harus habis dibagi num_heads."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.dropout = dropout
