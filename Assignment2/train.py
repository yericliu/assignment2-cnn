import argparse
import json
from pathlib import Path

import torch
from torch import nn

from helper_lib.data_loader import CLASSES, get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model, evaluate_model

BASE = Path(__file__).resolve().parent


def run_training(epochs=10, batch_size=64, learning_rate=0.001):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_loader = get_data_loader(BASE / "data", batch_size=batch_size)
    test_loader = get_data_loader(BASE / "data", batch_size=batch_size, train=False)
    model = get_model("CNN").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, train_loader, criterion, optimizer, device, epochs)
    results = evaluate_model(model, test_loader, criterion, device)
    results.update({"epochs": epochs, "batch_size": batch_size,
                    "learning_rate": learning_rate, "seed": 42,
                    "train_samples": len(train_loader.dataset),
                    "history": model.training_history})
    output_dir = BASE / "outputs"
    output_dir.mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "classes": CLASSES},
               output_dir / "cifar10_cnn.pth")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Test accuracy: {results['test_accuracy']:.2%}")
    print("Saved model and metrics to", output_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_training(epochs=args.epochs)
