from collections import Counter
from pathlib import Path
import json

CURRENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_PATH.parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
IWSLT14_DIR  = DATASETS_DIR / "iwslt14"

def build_vocab(path):
    counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    token_to_idx = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    for token, count in counter.most_common():
        if count >= 1 and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    idx_to_token = list(token_to_idx.keys())

    return token_to_idx, idx_to_token

# Create vocabularies
en_token_to_idx, en_idx_to_token = build_vocab(IWSLT14_DIR / "train.en")
de_token_to_idx, de_idx_to_token = build_vocab(IWSLT14_DIR / "train.de")

# Save vocabularies
en_token_to_idx_path = ARTIFACTS_DIR / "en_token_to_idx.json"
en_idx_to_token_path = ARTIFACTS_DIR / "en_idx_to_token.json"
de_token_to_idx_path = ARTIFACTS_DIR / "de_token_to_idx.json"
de_idx_to_token_path = ARTIFACTS_DIR / "de_idx_to_token.json"

with open(en_token_to_idx_path, "w", encoding="utf-8") as f:
    json.dump(en_token_to_idx, f, ensure_ascii=False, indent=2)
with open(en_idx_to_token_path, "w", encoding="utf-8") as f:
    json.dump(en_idx_to_token, f, ensure_ascii=False, indent=2)
with open(de_token_to_idx_path, "w", encoding="utf-8") as f:
    json.dump(de_token_to_idx, f, ensure_ascii=False, indent=2)
with open(de_idx_to_token_path, "w", encoding="utf-8") as f:
    json.dump(de_idx_to_token, f, ensure_ascii=False, indent=2)