from pathlib import Path

import torch
import torch.nn as nn

from training.load_data import PAD_IDX, create_dataloaders
from training.training_utils import (
    create_model,
    evaluate,
    get_device,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


CURRENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_PATH.parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


BATCH_SIZE = 64
MAX_LEN = 256
EPOCHS = 10
LEARNING_RATE = 1e-4


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocabs = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        vocabs=vocabs,
        device=device,
        max_len=MAX_LEN,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    checkpoint_path = ARTIFACTS_DIR / "best_translation_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                path=checkpoint_path,
            )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

    best_checkpoint = load_checkpoint(
        model=model,
        path=checkpoint_path,
        device=device,
    )

    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Best epoch: {best_checkpoint['epoch']}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
