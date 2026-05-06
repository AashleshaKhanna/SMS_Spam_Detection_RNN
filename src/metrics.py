"""Evaluation metrics for spam detection."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device | str) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        preds = outputs.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return correct / total


@torch.no_grad()
def false_positive_negative_rates(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    """Return false-positive rate and false-negative rate.

    Positive class = spam.
    False positive: predicted spam but true label is ham.
    False negative: predicted ham but true label is spam.
    """
    model.eval()

    false_positives = 0
    false_negatives = 0
    total_negatives = 0
    total_positives = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        preds = outputs.argmax(dim=1)

        for pred, label in zip(preds, labels):
            if label.item() == 0:
                total_negatives += 1
                if pred.item() == 1:
                    false_positives += 1
            else:
                total_positives += 1
                if pred.item() == 0:
                    false_negatives += 1

    fpr = false_positives / total_negatives if total_negatives else 0.0
    fnr = false_negatives / total_positives if total_positives else 0.0
    return fpr, fnr
