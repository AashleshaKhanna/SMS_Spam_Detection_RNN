"""Data loading and batching for SMS spam detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


@dataclass
class SpamData:
    train_sequences: list[list[int]]
    train_labels: list[int]
    val_sequences: list[list[int]]
    val_labels: list[int]
    test_sequences: list[list[int]]
    test_labels: list[int]
    stoi: dict[str, int]
    itos: dict[int, str]


class SMSDataset(Dataset):
    """Dataset for encoded SMS sequences."""

    def __init__(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


def collate_sequences(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length SMS character sequences in a batch."""
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)
    return padded_sequences, labels_tensor


def load_sms_file(path: str | Path) -> tuple[list[str], list[int]]:
    """Load SMS Spam Collection file.

    Returns raw text messages and labels, where ham=0 and spam=1.
    """
    texts: list[str] = []
    labels: list[int] = []

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue

            label, text = parts
            labels.append(1 if label == "spam" else 0)
            texts.append(text)

    return texts, labels


def build_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build character vocabulary. Index 0 is reserved for padding."""
    all_text = "".join(texts)
    chars = sorted(set(all_text))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    itos[0] = "<pad>"
    return stoi, itos


def encode_texts(texts: list[str], stoi: dict[str, int]) -> list[list[int]]:
    """Convert each SMS message to character IDs."""
    return [[stoi[ch] for ch in text if ch in stoi] for text in texts]


def balance_training_set(
    train_sequences: list[list[int]],
    train_labels: list[int],
    spam_multiplier: int = 7,
) -> tuple[list[list[int]], list[int]]:
    """Oversample spam examples to reduce training-set class imbalance.

    The lab duplicated each spam message 6 additional times, so each spam
    message appears 7 times total in the balanced training set.
    """
    balanced_sequences = list(train_sequences)
    balanced_labels = list(train_labels)

    for sequence, label in zip(train_sequences, train_labels):
        if label == 1:
            for _ in range(spam_multiplier - 1):
                balanced_sequences.append(sequence)
                balanced_labels.append(label)

    return balanced_sequences, balanced_labels


def prepare_data(
    data_path: str,
    test_size: float = 0.40,
    val_size_within_temp: float = 0.50,
    seed: int = 42,
    balance_train: bool = True,
) -> SpamData:
    """Load, encode, split, and optionally balance SMS spam data.

    First split: 60% train, 40% temp.
    Second split: temp into 20% validation and 20% test overall.
    """
    texts, labels = load_sms_file(data_path)
    stoi, itos = build_vocab(texts)
    sequences = encode_texts(texts, stoi)

    indices = list(range(len(sequences)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=val_size_within_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_sequences = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    if balance_train:
        train_sequences, train_labels = balance_training_set(train_sequences, train_labels)

    return SpamData(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
        stoi=stoi,
        itos=itos,
    )


def make_loaders(data: SpamData, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/validation/test splits."""
    train_loader = DataLoader(
        SMSDataset(data.train_sequences, data.train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        SMSDataset(data.val_sequences, data.val_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        SMSDataset(data.test_sequences, data.test_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    return train_loader, val_loader, test_loader
