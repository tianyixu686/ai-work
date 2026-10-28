"""Train and evaluate AnimalLiteNet."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import CLASSES, loaders
from model import AnimalLiteNet


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        count += y.size(0)
    return total_loss / count, correct / count


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader = loaders(args.data, args.batch_size, args.seed, args.workers)
    model = AnimalLiteNet(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best, stale, rows = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        rows.append([epoch, tr_loss, tr_acc, va_loss, va_acc, optimizer.param_groups[0]["lr"]])
        print(f"{epoch:02d} train={tr_acc:.4f} val={va_acc:.4f}")
        if va_acc > best:
            best, stale = va_acc, 0
            torch.save({"model": model.state_dict(), "classes": CLASSES, "val_accuracy": best}, out / "best_model.pt")
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]); w.writerows(rows)
    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    metrics = {"device": str(device), "best_val_accuracy": best, "test_loss": test_loss,
               "test_accuracy": test_acc, "epochs_ran": len(rows), "seed": args.seed}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data"); p.add_argument("--output", default="results")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026); p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())

