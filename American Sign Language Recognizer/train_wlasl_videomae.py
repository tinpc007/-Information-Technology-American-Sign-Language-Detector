# =========================
# train_wlasl_videomae_full.py
# =========================

import os
import json
import time
import random
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from decord import VideoReader, cpu

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    get_cosine_schedule_with_warmup,
)

# =========================
# Utils
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_label_maps(train_dir):
    classes = sorted([d.name for d in Path(train_dir).iterdir() if d.is_dir()])
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label

def get_samples(split_dir, label2id):
    samples = []
    for c, idx in label2id.items():
        p = Path(split_dir) / c
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.suffix in [".mp4", ".avi", ".mov"]:
                samples.append((str(f), idx))
    return samples

def uniform_sampling(n_frames, total):
    if total >= n_frames:
        return np.linspace(0, total - 1, n_frames).astype(int)
    idx = np.arange(total)
    pad = np.full(n_frames - total, total - 1)
    return np.concatenate([idx, pad])

# =========================
# Dataset
# =========================

class WLASLDataset(Dataset):
    def __init__(self, root, processor, label2id, num_frames=16, mode="train"):
        self.samples = get_samples(root, label2id)
        self.processor = processor
        self.num_frames = num_frames
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)

        idxs = uniform_sampling(self.num_frames, total)
        frames = vr.get_batch(idxs).asnumpy()

        frames = [frames[i] for i in range(len(frames))]
        inputs = self.processor(frames, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label)
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch])
    }

# =========================
# Early stopping
# =========================

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best = None

    def step(self, val):
        if self.best is None or val > self.best:
            self.best = val
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# =========================
# Train / Eval
# =========================

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_acc = 0, 0

    for batch in tqdm(loader, desc="Train"):
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        out = model(pixel_values=x, labels=y)
        loss = out.loss
        logits = out.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = (logits.argmax(1) == y).float().mean().item()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="Eval"):
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        out = model(pixel_values=x)
        all_logits.append(out.logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    top1 = top_k_accuracy_score(labels, logits, k=1)
    top5 = top_k_accuracy_score(labels, logits, k=min(5, logits.shape[1]))

    return top1, top5

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/haole/data/khanh/projects/gesture_recognition/datasets/wlasl-2000/wlasl-complete/wlasl100_clean")
    parser.add_argument("--output_dir", default="./runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    test_dir = os.path.join(args.data_root, "test")

    label2id, id2label = build_label_maps(train_dir)

    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=len(label2id),
        ignore_mismatched_sizes=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = WLASLDataset(train_dir, processor, label2id)
    val_ds = WLASLDataset(val_dir, processor, label2id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 100, args.epochs * len(train_loader)
    )

    early = EarlyStopping(patience=20)
    best_acc = 0

    log_csv = os.path.join(args.output_dir, "train_log.csv")

    with open(log_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_top1", "val_top5"])

    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        val_top1, val_top5 = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        print(
            f"[{epoch}] "
            f"TrainLoss {train_loss:.4f} | "
            f"TrainAcc {train_acc:.4f} || "
            f"ValTop1 {val_top1:.4f} | ValTop5 {val_top5:.4f} || "
            f"{elapsed:.1f}s"
        )

        # Save CSV
        with open(log_csv, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_top1, val_top5])

        # Save last
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))

        # Save best
        if val_top1 > best_acc:
            best_acc = val_top1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            print("🔥 Best model updated")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_top1": val_top1,
            "val_top5": val_top5
        })

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Early stop
        if early.step(val_top1):
            print("⛔ Early stopping triggered")
            break

    print("Training complete.")

if __name__ == "__main__":
    main()