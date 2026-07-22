import torch.nn as nn
import math
import torch

# batch size: B
# source sequence length (input): S
# target sequence length: T
# d_model: D
# vocabulary: V

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):          # [B, S] or [B, T]
        output = self.embedding(tokens) # [B, S, D] or [B, T, D]
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, d_model]
        # ->
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):       # [B, S, D] or [B, T, D]
        seq_len = x.size(1)
        output = x + self.pe[:, :seq_len, :]  # [B, S, D] or [B, T, D]
        return output


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            num_layers,
            dim_feedforward=2048,
            dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()

        # One encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack N identical encoder feature
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,                  # [B, S, D]
        src_key_padding_mask: torch.Tensor, # [B, S]
    ) -> torch.Tensor:

        output = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask
        )                                   # [B, S, D]
        return output

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # One decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Stack N decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

    def forward(
            self,
            tgt: torch.Tensor,                                      # [B, T, D]
            memory: torch.Tensor,                                   # [B, S, D]
            tgt_mask: torch.Tensor | None = None,                   # [T, T]
            tgt_key_padding_mask: torch.Tensor | None = None,       # [B, T]
            memory_key_padding_mask: torch.Tensor | None = None,    # [B, S]
    ) -> torch.Tensor:

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output

class OutputProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
        )

    def forward(
        self,
        input: torch.Tensor,        # [B, T, D]
    ) -> torch.Tensor:
        output = self.linear(input) # [B, T, V]
        return output