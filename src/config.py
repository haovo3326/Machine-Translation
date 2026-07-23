from pathlib import Path

CURRENT_PATH = Path(__file__).resolve()

# Project paths
PROJECT_ROOT = CURRENT_PATH.parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VOCABS_DIR = ARTIFACTS_DIR / "vocabs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
IWSLT14_DIR = DATASETS_DIR / "iwslt14"


# Dataset settings
SOURCE_LANGUAGE = "de"
TARGET_LANGUAGE = "en"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


# Special token ids
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


# Training settings
BATCH_SIZE = 64
MAX_LEN = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = True
CHECKPOINT_FILENAME = "best_translation_model.pt"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"
TEXT_ENCODING = "utf-8"
INITIAL_BEST_VAL_LOSS = float("inf")


# Model settings
D_MODEL = 256
N_HEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
