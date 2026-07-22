from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Dataset


CURRENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_PATH.parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "iwslt14"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(language):
    token_to_idx_path = ARTIFACTS_DIR / f"{language}_token_to_idx.json"
    idx_to_token_path = ARTIFACTS_DIR / f"{language}_idx_to_token.json"

    token_to_idx = load_json(token_to_idx_path)
    idx_to_token = load_json(idx_to_token_path)

    return token_to_idx, idx_to_token


class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_len=256):
        self.src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
        self.tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError(
                f"source and target files are not aligned: "
                f"{src_path} has {len(self.src_lines)} lines, "
                f"{tgt_path} has {len(self.tgt_lines)} lines"
            )

    def encode(self, line, vocab):
        tokens = line.strip().split()
        ids = [BOS_IDX]
        ids.extend(vocab.get(token, UNK_IDX) for token in tokens)
        ids.append(EOS_IDX)
        return ids[:self.max_len]

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.encode(self.src_lines[idx], self.src_vocab)
        tgt = self.encode(self.tgt_lines[idx], self.tgt_vocab)

        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long),
        )


def pad_batch(sequences):
    return torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )


def collate_fn(batch):
    src, tgt_input, tgt_output = zip(*batch)

    src = pad_batch(src)
    tgt_input = pad_batch(tgt_input)
    tgt_output = pad_batch(tgt_output)

    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "src_key_padding_mask": src == PAD_IDX,
        "tgt_key_padding_mask": tgt_input == PAD_IDX,
    }


def create_dataset(split, src_vocab, tgt_vocab, max_len=256):
    return TranslationDataset(
        DATA_DIR / f"{split}.de",
        DATA_DIR / f"{split}.en",
        src_vocab,
        tgt_vocab,
        max_len=max_len,
    )


def create_dataloader(
    split,
    src_vocab,
    tgt_vocab,
    batch_size=64,
    shuffle=False,
    max_len=256,
    num_workers=2,
    pin_memory=True,
):
    dataset = create_dataset(
        split=split,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_dataloaders(batch_size=64, max_len=256, num_workers=2, pin_memory=True):
    de_token_to_idx, de_idx_to_token = load_vocab("de")
    en_token_to_idx, en_idx_to_token = load_vocab("en")

    train_loader = create_dataloader(
        split="train",
        src_vocab=de_token_to_idx,
        tgt_vocab=en_token_to_idx,
        batch_size=batch_size,
        shuffle=True,
        max_len=max_len,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        split="val",
        src_vocab=de_token_to_idx,
        tgt_vocab=en_token_to_idx,
        batch_size=batch_size,
        shuffle=False,
        max_len=max_len,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = create_dataloader(
        split="test",
        src_vocab=de_token_to_idx,
        tgt_vocab=en_token_to_idx,
        batch_size=batch_size,
        shuffle=False,
        max_len=max_len,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    vocabs = {
        "src_token_to_idx": de_token_to_idx,
        "src_idx_to_token": de_idx_to_token,
        "tgt_token_to_idx": en_token_to_idx,
        "tgt_idx_to_token": en_idx_to_token,
    }

    return train_loader, val_loader, test_loader, vocabs
