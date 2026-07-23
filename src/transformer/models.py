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

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        bos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 256,
    ) -> list[list[int]]:
        batch_size = src.size(0)
        device = src.device
        decoded = [[] for _ in range(batch_size)]
        active_indices = torch.arange(batch_size, device=device)
        active_src = src
        active_src_key_padding_mask = src_key_padding_mask
        active_tgt = torch.full(
            (batch_size, 1),
            bos_idx,
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_len - 1):
            tgt_len = active_tgt.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            logits = self(
                src=active_src,
                tgt=active_tgt,
                src_key_padding_mask=active_src_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)
            for sequence_idx, token_id in zip(
                active_indices.tolist(),
                next_token.tolist(),
            ):
                decoded[sequence_idx].append(token_id)

            keep_mask = next_token.ne(eos_idx)
            if not keep_mask.any():
                break

            active_indices = active_indices[keep_mask]
            active_src = active_src[keep_mask]
            if active_src_key_padding_mask is not None:
                active_src_key_padding_mask = active_src_key_padding_mask[keep_mask]
            active_tgt = torch.cat(
                [active_tgt, next_token.unsqueeze(1)],
                dim=1,
            )[keep_mask]

        return decoded
