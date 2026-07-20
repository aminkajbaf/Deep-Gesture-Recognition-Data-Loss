#!/usr/bin/env python3
"""
Two-phase training for DGRDL (paper Sec. III-E).

Phase 1: train encoder-decoder with MSE on corrupted -> clean reconstruction.
Phase 2: freeze encoder; train Scale-Shift + classifier with cross-entropy.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dgrdl.data import create_dataloaders
from dgrdl.models import DGRDLModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict, dl_ratio: float):
    data = cfg["data"]
    return create_dataloaders(
        root_dirs=data["root_dirs"],
        negative_root=data.get("negative_root"),
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        seed=cfg["train"]["seed"],
        num_workers=data["num_workers"],
        image_size=tuple(data["image_size"]),
        dl_ratio=dl_ratio,
    )


@torch.no_grad()
def eval_reconstruction(model, loader, device, criterion):
    model.eval()
    total, n = 0.0, 0
    for clean, corrupt, _ in loader:
        clean, corrupt = clean.to(device), corrupt.to(device)
        recon, _ = model.reconstruct(corrupt)
        total += criterion(recon, clean).item() * clean.size(0)
        n += clean.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_classification(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for clean, corrupt, labels in loader:
        corrupt, labels = corrupt.to(device), labels.to(device)
        logits = model.classify(corrupt, labels=None)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return 100.0 * correct / max(n, 1)


def train_phase1(model, train_loader, val_loader, cfg, device):
    epochs = cfg["train"]["phase1_epochs"]
    lr = cfg["train"]["learning_rate"]
    ckpt = Path(cfg["paths"]["phase1_ckpt"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    best = float("inf")

    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Phase1 {epoch+1}/{epochs}")
        for clean, corrupt, _ in pbar:
            clean, corrupt = clean.to(device), corrupt.to(device)
            optimizer.zero_grad()
            recon, _ = model.reconstruct(corrupt)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()
            running += loss.item() * clean.size(0)
            n += clean.size(0)
            pbar.set_postfix(mse=f"{running / n:.6f}")

        val_mse = eval_reconstruction(model, val_loader, device, criterion)
        scheduler.step(val_mse)
        print(f"Phase1 epoch {epoch+1}: train_mse={running/n:.6f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            torch.save({"model": model.state_dict(), "val_mse": best}, ckpt)
            print(f"  saved {ckpt}")
    return ckpt


def train_phase2(model, train_loader, val_loader, cfg, device):
    epochs = cfg["train"]["phase2_epochs"]
    lr = cfg["train"]["learning_rate"]
    ckpt = Path(cfg["paths"]["phase2_ckpt"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model.freeze_encoder()
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=5)
    best = 0.0

    for epoch in range(epochs):
        model.train()
        # Keep backbone in eval for frozen BN stats; classifier stays in train
        model.backbone.eval()
        running, correct, n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Phase2 {epoch+1}/{epochs}")
        for clean, corrupt, labels in pbar:
            corrupt, labels = corrupt.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classify(corrupt, labels=labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
            pbar.set_postfix(loss=f"{running/n:.4f}", acc=f"{100*correct/n:.2f}%")

        val_acc = eval_classification(model, val_loader, device)
        scheduler.step(val_acc)
        print(
            f"Phase2 epoch {epoch+1}: "
            f"train_loss={running/n:.4f} train_acc={100*correct/n:.2f}% val_acc={val_acc:.2f}%"
        )
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best}, ckpt)
            print(f"  saved {ckpt} (best val acc {best:.2f}%)")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Train DGRDL (two-phase)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--dl-ratio", type=float, default=0.2, help="Simulated data-loss ratio")
    parser.add_argument("--resume-phase1", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = build_loaders(cfg, dl_ratio=args.dl_ratio)
    model = DGRDLModel(
        num_classes=cfg["data"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        depth_per_stage=cfg["model"]["depth_per_stage"],
        bottleneck_depth=cfg["model"]["bottleneck_depth"],
        scale_shift_hidden=cfg["model"]["scale_shift_hidden"],
    ).to(device)

    if args.phase in ("1", "all"):
        train_phase1(model, train_loader, val_loader, cfg, device)

    if args.phase in ("2", "all"):
        phase1_path = args.resume_phase1 or cfg["paths"]["phase1_ckpt"]
        if Path(phase1_path).is_file():
            state = torch.load(phase1_path, map_location=device)
            model.load_state_dict(state["model"], strict=False)
            print(f"Loaded Phase-1 checkpoint: {phase1_path}")
        elif args.phase == "2":
            raise FileNotFoundError(f"Phase-1 checkpoint not found: {phase1_path}")
        train_phase2(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
