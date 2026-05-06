"""Evaluate trained SMS spam RNN on held-out test set."""

from __future__ import annotations

import argparse
import torch

from src.data import prepare_data, make_loaders
from src.metrics import accuracy, false_positive_negative_rates
from src.model import SpamRNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SMS spam classifier.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_spam_rnn.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    spam_data = prepare_data(args.data_path, balance_train=True)
    _, _, test_loader = make_loaders(spam_data, batch_size=args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["metadata"]

    model = SpamRNN(
        vocab_size=metadata["vocab_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    test_acc = accuracy(model, test_loader, device)
    fpr, fnr = false_positive_negative_rates(model, test_loader, device)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"False positive rate: {fpr:.4f}")
    print(f"False negative rate: {fnr:.4f}")


if __name__ == "__main__":
    main()
