"""Train character-level RNN for SMS spam detection."""

from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn

from src.data import prepare_data, make_loaders
from src.metrics import accuracy
from src.model import SpamRNN
from src.utils import ensure_dirs, save_checkpoint, save_vocab


def epoch_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device | str,
) -> float:
    """Compute average loss over a loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples


def train(
    data_path: str,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 1,
    device: str | None = None,
) -> dict[str, list[float]]:
    ensure_dirs()
    torch.manual_seed(42)

    spam_data = prepare_data(data_path, balance_train=True)
    train_loader, val_loader, _ = make_loaders(spam_data, batch_size=batch_size)

    vocab_size = len(spam_data.stoi) + 1
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = SpamRNN(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers).to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        for sequences, labels in train_loader:
            sequences = sequences.to(device_obj)
            labels = labels.to(device_obj)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss(model, train_loader, criterion, device_obj)
        val_loss = epoch_loss(model, val_loader, criterion, device_obj)
        train_acc = accuracy(model, train_loader, device_obj)
        val_acc = accuracy(model, val_loader, device_obj)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                "checkpoints/best_spam_rnn.pt",
                model,
                {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "best_val_acc": best_val_acc,
                },
            )

    np.savetxt(
        "results/training_history.csv",
        np.column_stack(
            [
                history["train_loss"],
                history["val_loss"],
                history["train_acc"],
                history["val_acc"],
            ]
        ),
        delimiter=",",
        header="train_loss,val_loss,train_acc,val_acc",
        comments="",
    )

    save_vocab("checkpoints/vocab.json", spam_data.stoi, spam_data.itos)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMS spam character-level RNN.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
