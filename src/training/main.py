from transformer.models import EncoderDecoderModel
from pathlib import Path
import json


CURRENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_PATH.parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Load vocabularies
with open(ARTIFACTS_DIR / "en_token_to_idx.json", "r", encoding = "utf-8") as f:
    en_token_to_idx = json.load(f)
with open(ARTIFACTS_DIR / "de_token_to_idx.json", "r", encoding = "utf-8") as f:
    de_token_to_idx = json.load(f)

# Initialize model
model = EncoderDecoderModel(
    src_vocab_size=len(en_token_to_idx),
    tgt_vocab_size=len(de_token_to_idx),
    d_model=256,
    n_head=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=256
)