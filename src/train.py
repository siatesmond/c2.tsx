"""
train.py

Trains the AIGC detector: loads data via AIGCDataset with the robustness
augmentation pipeline, trains the model from model.py, saves checkpoints,
and tracks train/val loss + accuracy per epoch.

Usage:
    python3 -m src.train --data_dir data/train --epochs 5 --batch_size 32

    # quick smoke test on a tiny subset first (recommended before a full run):
    python3 -m src.train --data_dir data/train --epochs 1 --limit 200 --batch_size 16
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import AIGCDataset, get_train_transform, get_val_transform
from src.model import build_model, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AIGC detector")
    parser.add_argument("--data_dir", type=str, default="data/train",
                         help="Path to training data (must contain REAL/ and FAKE/ subfolders)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1,
                         help="Fraction of data_dir held out for validation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                         help="If set, only use this many samples total (for a fast smoke test)")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Skip loading pretrained weights (for offline smoke tests)")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # shape (batch, 1) to match model output

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=loss.item())

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    # Load once with train transform, then split into train/val subsets.
    # Note: val subset technically still uses train-time augmentation this way,
    # which is a known simplification for speed. For a fully clean val signal,
    # load a second AIGCDataset with get_val_transform() pointed at the same
    # files, or better: use data/test (the CIFAKE-provided held-out split) for
    # your real "clean" validation via evaluate.py instead.
    full_dataset = AIGCDataset(args.data_dir, transform=get_train_transform())

    if args.limit:
        from torch.utils.data import Subset
        import random
        random.seed(args.seed)
        indices = random.sample(range(len(full_dataset)), min(args.limit, len(full_dataset)))
        full_dataset = Subset(full_dataset, indices)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # --- Model ---
    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    print(f"Model: {args.backbone} ({count_parameters(model):,} parameters)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        # Save "last" checkpoint every epoch, and "best" whenever val_acc improves.
        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  -> New best val_acc={val_acc:.4f}, saved to {checkpoint_dir / 'best.pt'}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")


if __name__ == "__main__":
    main()