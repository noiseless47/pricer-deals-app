"""Evaluate the saved DNN checkpoint on the untouched test split.

Run from the repository root with:
    uv run --extra data python backend/scripts/evaluate_dnn.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.deep_neural_network import (  # noqa: E402
    DeepNeuralNetwork,
    Y_MEAN,
    Y_STD,
)


DATASET_NAME = "ed-donner/items_full"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/deep_neural_network.pth"
FEATURE_COUNT = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DNN evaluation requires CUDA; no CPU fallback is used.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    device = torch.device("cuda:0")
    print(f"Evaluating on {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading test split from {DATASET_NAME}...", flush=True)
    test = load_dataset(DATASET_NAME, split="test")

    vectorizer = HashingVectorizer(
        n_features=FEATURE_COUNT,
        stop_words="english",
        binary=True,
        dtype=np.float32,
    )
    print(f"Vectorizing {len(test):,} test rows...", flush=True)
    features = vectorizer.transform(test["summary"]).toarray()
    actual_prices = np.asarray(test["price"], dtype=np.float32)
    feature_tensor = torch.from_numpy(features)
    del features, test

    loader = DataLoader(
        TensorDataset(feature_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    del feature_tensor

    model = DeepNeuralNetwork(FEATURE_COUNT).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            normalized = model(batch).squeeze(1)
            prices = torch.expm1(normalized * Y_STD + Y_MEAN).clamp_min(0)
            predictions.append(prices.cpu().numpy())

    predicted_prices = np.concatenate(predictions)
    mae = float(np.mean(np.abs(predicted_prices - actual_prices)))
    train_prices = np.asarray(
        load_dataset(DATASET_NAME, split="train")["price"], dtype=np.float32
    )
    mean_price_baseline = float(np.mean(np.abs(np.mean(train_prices) - actual_prices)))
    median_price_baseline = float(np.mean(np.abs(np.median(train_prices) - actual_prices)))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test rows: {len(actual_prices):,}")
    print(f"DNN test MAE: ${mae:.2f}")
    print(f"Constant-mean baseline MAE: ${mean_price_baseline:.2f}")
    print(f"Constant-median baseline MAE: ${median_price_baseline:.2f}")


if __name__ == "__main__":
    main()
