from collections import Counter
import json

from config import (
    BOS_IDX,
    EOS_IDX,
    IWSLT14_DIR,
    PAD_IDX,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    TEXT_ENCODING,
    TRAIN_SPLIT,
    UNK_IDX,
    VOCABS_DIR,
)

def build_vocab(path):
    counter = Counter()
    with open(path, "r", encoding=TEXT_ENCODING) as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    token_to_idx = {
        "<pad>": PAD_IDX,
        "<unk>": UNK_IDX,
        "<bos>": BOS_IDX,
        "<eos>": EOS_IDX,
    }
    for token, count in counter.most_common():
        if count >= 1 and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    idx_to_token = list(token_to_idx.keys())

    return token_to_idx, idx_to_token

# Create vocabularies
en_token_to_idx, en_idx_to_token = build_vocab(
    IWSLT14_DIR / f"{TRAIN_SPLIT}.{TARGET_LANGUAGE}"
)
de_token_to_idx, de_idx_to_token = build_vocab(
    IWSLT14_DIR / f"{TRAIN_SPLIT}.{SOURCE_LANGUAGE}"
)

# Save vocabularies
VOCABS_DIR.mkdir(parents=True, exist_ok=True)

en_token_to_idx_path = VOCABS_DIR / f"{TARGET_LANGUAGE}_token_to_idx.json"
en_idx_to_token_path = VOCABS_DIR / f"{TARGET_LANGUAGE}_idx_to_token.json"
de_token_to_idx_path = VOCABS_DIR / f"{SOURCE_LANGUAGE}_token_to_idx.json"
de_idx_to_token_path = VOCABS_DIR / f"{SOURCE_LANGUAGE}_idx_to_token.json"

with open(en_token_to_idx_path, "w", encoding=TEXT_ENCODING) as f:
    json.dump(en_token_to_idx, f, ensure_ascii=False, indent=2)
with open(en_idx_to_token_path, "w", encoding=TEXT_ENCODING) as f:
    json.dump(en_idx_to_token, f, ensure_ascii=False, indent=2)
with open(de_token_to_idx_path, "w", encoding=TEXT_ENCODING) as f:
    json.dump(de_token_to_idx, f, ensure_ascii=False, indent=2)
with open(de_idx_to_token_path, "w", encoding=TEXT_ENCODING) as f:
    json.dump(de_idx_to_token, f, ensure_ascii=False, indent=2)
