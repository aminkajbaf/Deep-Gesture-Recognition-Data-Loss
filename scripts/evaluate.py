#!/usr/bin/env python3
"""Evaluate a trained DGRDL checkpoint under one or more DL ratios."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dgrdl.data import GESTURE_CLASSES, create_dataloaders
from dgrdl.models import DGRDLModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dl-ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGRDLModel(
        num_classes=cfg["data"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        depth_per_stage=cfg["model"]["depth_per_stage"],
        bottleneck_depth=cfg["model"]["bottleneck_depth"],
        scale_shift_hidden=cfg["model"]["scale_shift_hidden"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    names = list(GESTURE_CLASSES.keys())
    for ratio in args.dl_ratios:
        _, val_loader = create_dataloaders(
            root_dirs=cfg["data"]["root_dirs"],
            negative_root=cfg["data"].get("negative_root"),
            batch_size=cfg["data"]["batch_size"],
            val_split=cfg["data"]["val_split"],
            seed=cfg["train"]["seed"],
            num_workers=cfg["data"]["num_workers"],
            image_size=tuple(cfg["data"]["image_size"]),
            dl_ratio=ratio,
        )
        preds, labels = [], []
        with torch.no_grad():
            for _, corrupt, y in tqdm(val_loader, desc=f"DL={ratio}"):
                logits = model.classify(corrupt.to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        acc = 100.0 * sum(p == t for p, t in zip(preds, labels)) / max(len(labels), 1)
        print(f"\n=== DL ratio {ratio:.2f} | Acc {acc:.2f}% ===")
        print(classification_report(labels, preds, target_names=names[: cfg["data"]["num_classes"]], digits=4))
        print("Confusion matrix:\n", confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
