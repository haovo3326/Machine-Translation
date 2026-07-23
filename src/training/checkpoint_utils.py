import json

import torch

from config import (
    CHECKPOINTS_DIR,
    CHECKPOINT_FILENAME,
    TEXT_ENCODING,
    TRAIN_DIR_PREFIX,
    TRAIN_LOSS_FILENAME,
    VAL_LOSS_FILENAME,
)


CURRENT_TRAIN_DIR = None
train_loss_logs = []
val_loss_logs = []


def create_train_dir():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    train_numbers = []
    for path in CHECKPOINTS_DIR.iterdir():
        if path.is_dir() and path.name.startswith(TRAIN_DIR_PREFIX):
            train_number = path.name.removeprefix(TRAIN_DIR_PREFIX)
            if train_number.isdigit():
                train_numbers.append(int(train_number))

    next_train_number = 0
    if train_numbers:
        next_train_number = max(train_numbers) + 1

    train_dir = CHECKPOINTS_DIR / f"{TRAIN_DIR_PREFIX}{next_train_number}"
    train_dir.mkdir(parents=True, exist_ok=False)

    return train_dir


def get_current_train_dir():
    global CURRENT_TRAIN_DIR

    if CURRENT_TRAIN_DIR is None:
        CURRENT_TRAIN_DIR = create_train_dir()

    return CURRENT_TRAIN_DIR


def get_checkpoint_path():
    return get_current_train_dir() / CHECKPOINT_FILENAME


def save_json(path, data):
    with open(path, "w", encoding=TEXT_ENCODING) as f:
        json.dump(data, f, indent=2)


def log_loss(epoch, train_loss, val_loss):
    train_dir = get_current_train_dir()

    train_loss_logs.append({
        "epoch": epoch,
        "loss": train_loss,
    })
    val_loss_logs.append({
        "epoch": epoch,
        "loss": val_loss,
    })

    save_json(train_dir / TRAIN_LOSS_FILENAME, train_loss_logs)
    save_json(train_dir / VAL_LOSS_FILENAME, val_loss_logs)


def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(checkpoint, get_checkpoint_path())


def load_checkpoint(model, device):
    checkpoint = torch.load(get_checkpoint_path(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint
