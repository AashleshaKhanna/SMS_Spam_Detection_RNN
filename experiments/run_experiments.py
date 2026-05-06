"""Run hyperparameter experiments for SMS spam RNN."""

from __future__ import annotations

import argparse
import pandas as pd

from src.train import train


EXPERIMENTS = [
    {"name": "base_gru64", "hidden_size": 64, "num_layers": 1, "learning_rate": 1e-3, "epochs": 15},
    {"name": "gru32", "hidden_size": 32, "num_layers": 1, "learning_rate": 1e-3, "epochs": 15},
    {"name": "gru128", "hidden_size": 128, "num_layers": 1, "learning_rate": 1e-3, "epochs": 15},
    {"name": "gru64_2layers", "hidden_size": 64, "num_layers": 2, "learning_rate": 1e-3, "epochs": 15},
    {"name": "gru64_low_lr", "hidden_size": 64, "num_layers": 1, "learning_rate": 5e-4, "epochs": 15},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMS spam RNN experiments.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    results = []

    for exp in EXPERIMENTS:
        print("=" * 80)
        print(f"Running {exp['name']}: {exp}")
        print("=" * 80)

        history = train(
            data_path=args.data_path,
            epochs=exp["epochs"],
            batch_size=args.batch_size,
            learning_rate=exp["learning_rate"],
            hidden_size=exp["hidden_size"],
            num_layers=exp["num_layers"],
        )

        results.append(
            {
                **exp,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
            }
        )

    df = pd.DataFrame(results).sort_values("final_val_acc", ascending=False)
    df.to_csv("results/experiment_summary.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
