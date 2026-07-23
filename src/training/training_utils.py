import torch
from transformer.models import EncoderDecoderModel


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }

def create_tgt_mask(tgt_len, device):
    return torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1,
    )

def create_model(vocabs, device, max_len):
    model = EncoderDecoderModel(
        src_vocab_size=len(vocabs["src_token_to_idx"]),
        tgt_vocab_size=len(vocabs["tgt_token_to_idx"]),
        d_model=256,
        n_head=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=max_len,
    )

    return model.to(device)

def calculate_loss(logits, tgt_output, criterion):
    return criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_output.reshape(-1),
    )


def forward_batch(model, batch):
    tgt_mask = create_tgt_mask(
        tgt_len=batch["tgt_input"].size(1),
        device=batch["tgt_input"].device,
    )

    return model(
        src=batch["src"],
        tgt=batch["tgt_input"],
        src_key_padding_mask=batch["src_key_padding_mask"],
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=batch["tgt_key_padding_mask"],
    )


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = forward_batch(model, batch)
        loss = calculate_loss(logits, batch["tgt_output"], criterion)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)

            logits = forward_batch(model, batch)
            loss = calculate_loss(logits, batch["tgt_output"], criterion)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint
