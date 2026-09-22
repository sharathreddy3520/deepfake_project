"""
train.py
Unified training entry-point for image and audio deepfake detectors.

Usage examples:
    # Train image model
    python train.py --modality image --data_dir data/images --epochs 10

    # Train audio model
    python train.py --modality audio --data_dir data/audio --epochs 20

    # Quick smoke-test with dummy data (no real dataset needed)
    python train.py --modality image --dummy --epochs 2
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.image_model   import build_image_model, save_model
from audio_processing.audio_model import build_audio_model, save_audio_model
from models.datasets      import make_image_loaders, make_audio_loaders
from utils.metrics        import compute_metrics, print_metrics, plot_training_history
from utils.logger         import get_logger

logger = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
#  Dummy data generators (for testing without a real dataset)
# ─────────────────────────────────────────────────────────────────────────────
def _make_dummy_image_loader(n: int = 32, batch_size: int = 8):
    """Return a DataLoader with random (3,224,224) tensors and binary labels."""
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n,)).float()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _make_dummy_audio_loader(n: int = 32, batch_size: int = 8):
    """Return a DataLoader with random (1,128,125) tensors and binary labels."""
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(n, 1, 128, 125)
    y = torch.randint(0, 2, (n,)).float()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-epoch train / validate helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="train" if training else "val ", leave=False):
            X, y = batch
            X = X.to(device)
            y = y.float().unsqueeze(1).to(device)

            logits = model(X)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs   = torch.sigmoid(logits).detach().cpu()
            preds   = (probs >= 0.5).long().squeeze(1)
            labels  = y.long().cpu().squeeze(1)

            total_loss += loss.item() * X.size(0)
            correct    += (preds == labels).sum().item()
            total      += X.size(0)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.squeeze(1).tolist())

    avg_loss = total_loss / max(total, 1)
    avg_acc  = correct   / max(total, 1)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, avg_acc, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"Modality: {args.modality}")

    # ── Build model ───────────────────────────────────────────────────────────
    if args.modality == "image":
        model = build_image_model(freeze_backbone=args.freeze_backbone).to(device)
    else:
        model = build_audio_model().to(device)

    logger.info(
        f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ── Data loaders ──────────────────────────────────────────────────────────
    if args.dummy:
        logger.info("Using DUMMY data (random tensors) for smoke-testing")
        if args.modality == "image":
            train_loader = _make_dummy_image_loader(64, args.batch_size)
            val_loader   = _make_dummy_image_loader(16, args.batch_size)
        else:
            train_loader = _make_dummy_audio_loader(64, args.batch_size)
            val_loader   = _make_dummy_audio_loader(16, args.batch_size)
    else:
        if args.modality == "image":
            train_loader, val_loader = make_image_loaders(
                args.data_dir,
                batch_size=args.batch_size,
                use_face_crop=args.face_crop,
            )
        else:
            train_loader, val_loader = make_audio_loaders(
                args.data_dir, batch_size=args.batch_size
            )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(
        args.save_dir,
        f"best_{args.modality}_model.pth"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_met = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        vl_loss, vl_acc, vl_met = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"| train loss={tr_loss:.4f} acc={tr_acc:.4f} "
            f"| val   loss={vl_loss:.4f} acc={vl_acc:.4f} "
            f"f1={vl_met['f1_score']:.4f} "
            f"| {time.time()-t0:.1f}s"
        )

        # Save best model
        if vl_met["f1_score"] >= best_val_f1:
            best_val_f1 = vl_met["f1_score"]
            if args.modality == "image":
                save_model(model, best_path)
            else:
                save_audio_model(model, best_path)
            logger.info(f"  → New best F1={best_val_f1:.4f}, model saved.")

    # ── Final report ─────────────────────────────────────────────────────────
    logger.info("Training complete.")
    print_metrics(vl_met)

    # Save training curves
    curve_path = os.path.join(args.save_dir, f"{args.modality}_training_curves.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          save_path=curve_path)
    logger.info(f"Training curves saved → {curve_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Detection – Training Script")
    p.add_argument("--modality", choices=["image", "audio"], required=True,
                   help="Which modality to train")
    p.add_argument("--data_dir", type=str, default="data/images",
                   help="Root data directory (must contain real/ and fake/ sub-dirs)")
    p.add_argument("--save_dir", type=str, default="models",
                   help="Directory to save model checkpoints")
    p.add_argument("--epochs",   type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone weights (image model only)")
    p.add_argument("--face_crop", action="store_true",
                   help="Detect and crop faces before training (image only)")
    p.add_argument("--cpu",  action="store_true", help="Force CPU training")
    p.add_argument("--dummy", action="store_true",
                   help="Use random dummy data (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
