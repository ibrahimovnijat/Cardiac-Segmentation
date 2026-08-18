"""Train the 2D U-Net baseline on CAMUS.

Loss is MONAI's ``DiceCELoss`` - soft Dice plus cross-entropy, the standard
pairing for medical segmentation. Dice supplies a gradient that cares about the
whole structure despite heavy class imbalance (background is ~78% of pixels),
while CE keeps per-pixel probabilities calibrated and stabilises early epochs
when Dice's gradient is nearly flat.

Usage::

    python train.py --epochs 50
    python train.py --epochs 1 --limit-train-batches 5 --limit-val-batches 2  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from camus_dataset import NUM_CLASSES, CAMUSDataset
from evaluation import SegmentationEvaluator
from model import build_unet, count_parameters, resolve_device


@dataclass
class TrainConfig:
    """Everything that defines a run, saved next to the checkpoint."""

    data_root: str
    out_dir: str = "runs/unet_baseline"
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    image_size: int = 256
    num_workers: int = 4
    seed: int = 42
    device: str | None = None
    amp: bool = False
    include_background_in_dice: bool = False
    lambda_dice: float = 1.0
    lambda_ce: float = 1.0
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    eval_distances_every: int = 0


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the official patient-level splits."""
    size = (cfg.image_size, cfg.image_size)
    train_ds = CAMUSDataset(cfg.data_root, split="train", image_size=size)
    val_ds = CAMUSDataset(cfg.data_root, split="val", image_size=size)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    common: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        # Pinned memory only helps host->CUDA transfers; MPS warns and ignores it.
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    limit_batches: int | None,
    epoch: int,
) -> float:
    model.train()
    total, seen = 0.0, 0
    use_amp = scaler is not None

    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for step, batch in enumerate(progress):
        if limit_batches is not None and step >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        # DiceCELoss with to_onehot_y expects the target as (B, 1, H, W).
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} at epoch {epoch} step {step}.")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item() * images.shape[0]
        seen += images.shape[0]
        progress.set_postfix(loss=f"{total / seen:.4f}")

    return total / max(seen, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    compute_distances: bool,
    limit_batches: int | None,
    epoch: int,
) -> tuple[float, dict[str, Any], SegmentationEvaluator]:
    model.eval()
    evaluator = SegmentationEvaluator(compute_distances=compute_distances)
    total, seen = 0.0, 0

    progress = tqdm(loader, desc=f"epoch {epoch} val", leave=False)
    for step, batch in enumerate(progress):
        if limit_batches is not None and step >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels.unsqueeze(1))

        total += loss.item() * images.shape[0]
        seen += images.shape[0]
        evaluator.update(
            logits=logits,
            labels=labels,
            spacing=batch["spacing"],
            meta={
                key: batch[key]
                for key in ("key", "patient", "view", "instant", "image_quality")
                if key in batch
            },
        )

    return total / max(seen, 1), evaluator.compute(), evaluator


def save_checkpoint(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AMP is a CUDA feature here; on MPS autocast is still rough, and on CPU it
    # buys nothing, so it stays off unless CUDA is present and it was asked for.
    use_amp = cfg.amp and device.type == "cuda"
    if cfg.amp and not use_amp:
        print(f"AMP requested but not enabled on device '{device.type}'; running in fp32.")

    train_loader, val_loader = build_data_loaders(cfg)
    model = build_unet().to(device)

    # Define DICE + Cross Entropy loss, with optional background inclusion and weighting.
    loss_fn = DiceCELoss(
        include_background=cfg.include_background_in_dice,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=cfg.lambda_dice,
        lambda_ce=cfg.lambda_ce,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    print(f"device      {device}")
    print(f"parameters  {count_parameters(model):,}")
    print(f"train       {len(train_loader.dataset)} samples / {len(train_loader)} batches")
    print(f"val         {len(val_loader.dataset)} samples / {len(val_loader)} batches")
    print(f"loss        DiceCE (dice={cfg.lambda_dice}, ce={cfg.lambda_ce}, "
          f"include_background={cfg.include_background_in_dice})\n")

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    history: list[dict[str, Any]] = []
    best_dice = -float("inf")

    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            cfg.limit_train_batches, epoch,
        )
        distances = bool(
            cfg.eval_distances_every and epoch % cfg.eval_distances_every == 0
        ) or epoch == cfg.epochs
        val_loss, summary, evaluator = validate(
            model, val_loader, loss_fn, device, distances, cfg.limit_val_batches, epoch
        )
        scheduler.step()

        val_dice = summary["mean_dice"]
        per_class = {n: round(summary["per_class"][n]["dice"], 4) for n in evaluator.class_names}
        elapsed = time.time() - started
        print(
            f"epoch {epoch:>3}/{cfg.epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_dice {val_dice:.4f}  {per_class}  "
            f"[{elapsed:.0f}s]"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": scheduler.get_last_lr()[0],
            "seconds": elapsed,
            "summary": summary,
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_dice": val_dice,
            "config": asdict(cfg),
            "torch_rng_state": torch.get_rng_state(),
        }
        save_checkpoint(out_dir / "last.pt", **state)

        # Selection is on the validation metric that matters, not on loss.
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(out_dir / "best.pt", **state)
            print(f"    new best val_dice {best_dice:.4f} -> {out_dir / 'best.pt'}")

    print(f"\nbest val_dice {best_dice:.4f}")
    print(f"artifacts in {out_dir.resolve()}")
    if history:
        print("\nfinal-epoch validation:")
        print(evaluator.report(history[-1]["summary"]))


def parse_args() -> TrainConfig:
    default_root = Path(__file__).resolve().parent.parent / "CAMUS_public"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(default_root))
    parser.add_argument("--out-dir", default="runs/unet_baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--include-background-in-dice", action="store_true")
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--eval-distances-every",
        type=int,
        default=0,
        help="compute HD95/ASSD every N epochs (0 = only on the final epoch)",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
