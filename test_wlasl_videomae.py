# =========================
# test_wlasl_videomae.py
# =========================

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import csv

import torch
from torch.utils.data import Dataset, DataLoader

from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

from sklearn.metrics import top_k_accuracy_score, confusion_matrix

# =========================
# Dataset
# =========================

class WLASLDataset(Dataset):
    def __init__(self, root, processor, label2id, num_frames=16):
        self.samples = []
        self.processor = processor
        self.num_frames = num_frames

        for cls, idx in label2id.items():
            cls_dir = os.path.join(root, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.endswith(".mp4"):
                    self.samples.append((os.path.join(cls_dir, f), idx))

    def __len__(self):
        return len(self.samples)

    def uniform_sample(self, total):
        if total >= self.num_frames:
            return np.linspace(0, total - 1, self.num_frames).astype(int)
        idx = np.arange(total)
        pad = np.full(self.num_frames - total, total - 1)
        return np.concatenate([idx, pad])

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)

        idxs = self.uniform_sample(total)
        frames = vr.get_batch(idxs).asnumpy()

        frames = [frames[i] for i in range(len(frames))]
        inputs = self.processor(frames, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label,
            "path": path
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch]),
        "paths": [b["path"] for b in batch]
    }

# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_paths = []

    for batch in tqdm(loader, desc="Testing"):
        x = batch["pixel_values"].to(device)
        y = batch["labels"]

        out = model(pixel_values=x)
        logits = out.logits.cpu().numpy()

        all_logits.append(logits)
        all_labels.append(y.numpy())
        all_paths.extend(batch["paths"])

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    preds = np.argmax(logits, axis=1)

    # Metrics
    top1 = top_k_accuracy_score(labels, logits, k=1)
    top5 = top_k_accuracy_score(labels, logits, k=min(5, logits.shape[1]))

    cm = confusion_matrix(labels, preds)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)

    return {
        "top1": float(top1),
        "top5": float(top5),
        "mean_class_acc": float(per_class_acc.mean()),
        "logits": logits,
        "labels": labels,
        "preds": preds,
        "paths": all_paths,
        "confusion_matrix": cm
    }

# =========================
# Main
# =========================
DATA_ROOT = "/home/haole/data/khanh/projects/gesture_recognition/datasets/wlasl-2000/wlasl-complete/wlasl100_clean"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--model_ckpt", default="runs/best.pth")
    parser.add_argument("--label_map", default=os.path.join(DATA_ROOT, "label2id.json"))
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load label map
    with open(args.label_map) as f:
        label2id = json.load(f)

    # Model
    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=len(label2id),
        ignore_mismatched_sizes=True
    )

    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Dataset
    test_dir = os.path.join(args.data_root, "test")

    dataset = WLASLDataset(test_dir, processor, label2id)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)

    # Evaluate
    results = evaluate(model, loader, device)

    # Save metrics
    metrics = {
        "top1": results["top1"],
        "top5": results["top5"],
        "mean_class_acc": results["mean_class_acc"]
    }

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"),
            results["confusion_matrix"])

    # Save predictions
    csv_path = os.path.join(args.output_dir, "predictions.csv")

    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "gt", "pred"])

        for p, gt, pred in zip(results["paths"],
                               results["labels"],
                               results["preds"]):
            writer.writerow([p, gt, pred])

    print("\n✅ TEST RESULTS")
    print(metrics)

# =========================

if __name__ == "__main__":
    main()