"""Train the local price-regression network on ed-donner/items_full.

Run from the repository root with:
    uv run --extra data python backend/scripts/train_dnn.py

The script requires CUDA and writes a compatible state_dict to
data/deep_neural_network.pth, which is the path used by NeuralNetworkAgent.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.deep_neural_network import (  # noqa: E402
    DeepNeuralNetwork,
    Y_MEAN,
    Y_STD,
)


DATASET_NAME = "ed-donner/items_full"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/deep_neural_network.pth"
FEATURE_COUNT = 5000


def make_tensors(dataset, vectorizer, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Vectorizing {label} split ({len(dataset):,} rows)...", flush=True)
    texts = dataset["summary"]
    prices = np.asarray(dataset["price"], dtype=np.float32)
    features = vectorizer.transform(texts).toarray()
    del texts

    log_prices = np.log1p(prices)
    targets = ((log_prices - Y_MEAN) / Y_STD).astype(np.float32).reshape(-1, 1)
    del prices, log_prices

    return torch.from_numpy(features), torch.from_numpy(targets)


@torch.inference_mode()
def validate(model, loader, device) -> tuple[float, float]:
    model.eval()
    normalized_loss = 0.0
    dollar_error = 0.0
    rows = 0

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(features)
        batch_rows = len(targets)

        normalized_loss += nn.functional.l1_loss(
            predictions, targets, reduction="sum"
        ).item()
        predicted_prices = torch.expm1(predictions * Y_STD + Y_MEAN).clamp_min(0)
        actual_prices = torch.expm1(targets * Y_STD + Y_MEAN).clamp_min(0)
        dollar_error += torch.abs(predicted_prices - actual_prices).sum().item()
        rows += batch_rows

    return normalized_loss / rows, dollar_error / rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DNN training requires a CUDA GPU; no CPU fallback is used.")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.weight_decay < 0
    ):
        parser.error("epochs and batch-size must be positive; learning-rate positive; weight-decay nonnegative")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    print(f"Training on {torch.cuda.get_device_name(device)}", flush=True)

    print(f"Loading {DATASET_NAME}...", flush=True)
    dataset = load_dataset(DATASET_NAME)
    vectorizer = HashingVectorizer(
        n_features=FEATURE_COUNT,
        stop_words="english",
        binary=True,
        dtype=np.float32,
    )

    train_features, train_targets = make_tensors(dataset["train"], vectorizer, "train")
    validation_features, validation_targets = make_tensors(
        dataset["validation"], vectorizer, "validation"
    )
    del dataset

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    del train_features, train_targets, validation_features, validation_targets

    model = DeepNeuralNetwork(FEATURE_COUNT).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_error = float("inf")
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rows = 0

        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = nn.functional.l1_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_rows = len(targets)
            total_loss += loss.detach().item() * batch_rows
            rows += batch_rows

        train_loss = total_loss / rows
        validation_loss, validation_mae = validate(model, validation_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_l1={train_loss:.5f}, "
            f"validation_l1={validation_loss:.5f}, "
            f"validation_mae=${validation_mae:.2f}",
            flush=True,
        )

        if validation_mae < best_validation_error:
            best_validation_error = validation_mae
            torch.save(model.state_dict(), temporary_output)
            os.replace(temporary_output, args.output)
            print(f"Saved best checkpoint to {args.output}", flush=True)

    print(
        f"Training complete. Best validation MAE: ${best_validation_error:.2f}. "
        f"Checkpoint: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
