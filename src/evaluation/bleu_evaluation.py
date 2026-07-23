import sacrebleu
import torch

from config import (
    BATCH_SIZE,
    BOS_IDX,
    CHECKPOINTS_DIR,
    CUDA_DEVICE,
    EOS_IDX,
    MAX_LEN,
    NUM_WORKERS,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
)
from training.load_data import create_dataloaders, load_vocab
from training.training_utils import create_model, get_device, move_batch_to_device


def load_checkpoint_model(path, device):
    src_token_to_idx, src_idx_to_token = load_vocab(SOURCE_LANGUAGE)
    tgt_token_to_idx, tgt_idx_to_token = load_vocab(TARGET_LANGUAGE)
    vocabs = {
        "src_token_to_idx": src_token_to_idx,
        "src_idx_to_token": src_idx_to_token,
        "tgt_token_to_idx": tgt_token_to_idx,
        "tgt_idx_to_token": tgt_idx_to_token,
    }

    model = create_model(
        vocabs=vocabs,
        device=device,
        max_len=MAX_LEN,
    )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def ids_to_text(ids, idx_to_token):
    tokens = []

    for token_id in ids:
        if token_id == EOS_IDX:
            break
        if token_id < 4:
            continue
        tokens.append(idx_to_token[token_id])

    return " ".join(tokens)


def bleu(model, data_loader, device):
    model.eval()
    _, tgt_idx_to_token = load_vocab(TARGET_LANGUAGE)
    predictions = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            print(f"Evaluating batch {batch_idx}/{len(data_loader)}")
            batch = move_batch_to_device(batch, device)
            predicts = model.greedy_decode(
                src=batch["src"],
                src_key_padding_mask=batch["src_key_padding_mask"],
                bos_idx=BOS_IDX,
                eos_idx=EOS_IDX,
                max_len=MAX_LEN,
            )

            predictions.extend(
                ids_to_text(predict, tgt_idx_to_token)
                for predict in predicts
            )
            references.extend(
                ids_to_text(ground_truth, tgt_idx_to_token)
                for ground_truth in batch["tgt_output"].tolist()
            )

    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU: {bleu_score.score:.2f}")

    return bleu_score


def main():
    device = get_device()
    print(f"Using device: {device}")
    checkpoint_path = CHECKPOINTS_DIR / "train0" / "best_translation_model.pt"
    assert checkpoint_path.exists()

    model = load_checkpoint_model(checkpoint_path, device)
    _, _, test_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == CUDA_DEVICE,
    )
    bleu(model, test_loader, device)


if __name__ == "__main__":
    main()
