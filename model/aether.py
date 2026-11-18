import torch
import torch.nn as nn

from model.config import AetherConfig


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            config.embedding_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            config.embedding_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.embedding_dim * 4
            ),
            nn.GELU(),
            nn.Linear(
                config.embedding_dim * 4,
                config.embedding_dim
            ),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        # Pre-normalization
        normalized = self.norm1(x)

        sequence_length = normalized.size(1)

        # Causal mask:
        # token tidak boleh melihat masa depan
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False
        )

        x = x + self.dropout(attention_output)

        # Feed forward
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class AetherModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.output = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.output.weight = (
            self.token_embedding.weight
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence terlalu panjang. "
                f"Maksimum: {self.config.context_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8
    ):

        self.eval()

        for _ in range(max_new_tokens):

            input_context = input_ids[
                :, -self.config.context_length:
            ]

            logits = self(input_context)

            logits = logits[:, -1, :]

            logits = logits / max(
                temperature,
                1e-6
            )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        return input_ids
