import torch
import torch.nn as nn

from .modules import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    TransformerDecoder,
    OutputProjection,
)

class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_head: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        # Source embedding
        self.src_embedding = TokenEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
        )

        # Target embedding
        self.tgt_embedding = TokenEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_head=n_head,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_head=n_head,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Decoder output -> target vocabulary logits
        self.output_projection = OutputProjection(
            d_model=d_model,
            vocab_size=tgt_vocab_size,
        )

    def forward(
        self,
        src: torch.Tensor,                                      # [B, S]
        tgt: torch.Tensor,                                      # [B, T]
        src_key_padding_mask: torch.Tensor | None = None,       # [B, S]
        tgt_mask: torch.Tensor | None = None,                   # [T, T]
        tgt_key_padding_mask: torch.Tensor | None = None,       # [B, T]
    ) -> torch.Tensor:

        # [B, S] -> [B, S, D]
        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        # [B, S, D] -> [B, S, D]
        memory = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
        )

        # [B, T] -> [B, T, D]
        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        # [B, T, D] -> [B, T, D]
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # [B, T, D] -> [B, T, V]
        logits = self.output_projection(output)

        return logits