from pathlib import Path

CURRENT_PATH = Path(__file__).resolve() # dataset_preprocessing folder
PROJECT_ROOT = CURRENT_PATH.parents[2]  # Translation Model folder
DATASET_DIR = PROJECT_ROOT / "datasets"
IWSLT14_DIR  = DATASET_DIR / "iwslt14"

with open(IWSLT14_DIR / "test.de", "r", encoding = "utf-8") as f:
    de_sentence = f.readlines()

with open(IWSLT14_DIR / "test.en", "r", encoding = "utf-8") as f:
    en_sentence = f.readlines()

for sentence in en_sentence:
    print(sentence)