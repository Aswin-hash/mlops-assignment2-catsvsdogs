"""Train the baseline Cats vs Dogs CNN with MLflow experiment tracking.

Usage:
    python -m src.models.train --params params.yaml
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.model import build_model

CLASS_NAMES = ["cat", "dog"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(processed_dir: Path, image_size: int, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(processed_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(processed_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(processed_dir / "test", transform=eval_tf)

    assert train_ds.classes == CLASS_NAMES, f"Unexpected class order: {train_ds.classes}"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_loss_curve(train_losses, val_losses, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, out_path: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (test set)")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    data_p, train_p, mlflow_p = params["data"], params["train"], params["mlflow"]
    set_seed(train_p["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_dir = Path(data_p["processed_dir"])

    train_loader, val_loader, test_loader = build_dataloaders(
        processed_dir, data_p["image_size"], train_p["batch_size"], train_p["num_workers"]
    )

    model = build_model(train_p["model"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_p["lr"])

    mlflow.set_tracking_uri(mlflow_p["tracking_uri"])
    mlflow.set_experiment(mlflow_p["experiment_name"])

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({**data_p, **train_p})

        train_losses, val_losses, val_accuracies = [], [], []
        for epoch in range(train_p["epochs"]):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            y_val_true, y_val_pred = evaluate(model, val_loader, device)
            val_acc = accuracy_score(y_val_true, y_val_pred)
            val_accuracies.append(val_acc)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )
            print(f"epoch {epoch+1}/{train_p['epochs']} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Final test-set evaluation
        y_test_true, y_test_pred = evaluate(model, test_loader, device)
        test_acc = accuracy_score(y_test_true, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_true, y_test_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_test_true, y_test_pred)

        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        })

        loss_curve_path = reports_dir / "loss_curve.png"
        cm_path = reports_dir / "confusion_matrix.png"
        plot_loss_curve(train_losses, val_losses, loss_curve_path)
        plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(str(loss_curve_path))
        mlflow.log_artifact(str(cm_path))

        model_path = models_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.pytorch.log_model(model, "model")

        metrics = {
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "val_accuracy_final": val_accuracies[-1] if val_accuracies else None,
        }
        with open(reports_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nSaved model to {model_path}")
        print(f"Test accuracy={test_acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
