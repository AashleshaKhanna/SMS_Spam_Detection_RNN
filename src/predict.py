"""Run inference for a single SMS message."""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from src.model import SpamRNN
from src.utils import load_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam probability for one SMS.")
    parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_spam_rnn.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stoi, _ = load_vocab(args.vocab)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["metadata"]

    model = SpamRNN(
        vocab_size=metadata["vocab_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids = [stoi[ch] for ch in args.message if ch in stoi]
    if not ids:
        raise ValueError("Message contains no characters from the training vocabulary.")

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        spam_probability = probs[0, 1].item()

    label = "spam" if spam_probability >= 0.5 else "ham"
    print(f"Prediction: {label}")
    print(f"Spam probability: {spam_probability:.6f}")


if __name__ == "__main__":
    main()
