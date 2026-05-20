"""Multi-label ILS interference classification training script.

Uses timm ResNet backbone with BCEWithLogitsLoss for 5-class multi-label
classification of spectrogram images (Arc, HV, IR, Jam, Lamp).

Example usage:
    python train_ils.py --model resnet18 --epochs 50 --batch-size 32
    python train_ils.py --model sigresnet18 --pretrained --lr 1e-3
"""
import os as _os
_os.environ.setdefault('HF_HUB_OFFLINE', '1')

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from dataset_ils import ILSInterferenceDataset, LABEL_NAMES, get_transforms


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor):
    """Compute multi-label metrics.

    Args:
        outputs: model logits, shape [N, 5]
        targets: binary labels, shape [N, 5]

    Returns:
        dict of metric_name -> float
    """
    probs = torch.sigmoid(outputs).cpu().numpy()
    preds = (probs >= 0.5).astype(np.float32)
    targets_np = targets.cpu().numpy()

    # Subset accuracy (exact match ratio)
    subset_acc = (preds == targets_np).all(axis=1).mean()

    # Hamming loss (lower is better)
    ham_loss = hamming_loss(targets_np, preds)

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[f'acc_{name}'] = (preds[:, i] == targets_np[:, i]).mean()
        per_class[f'prec_{name}'] = precision_score(targets_np[:, i], preds[:, i], zero_division=0)
        per_class[f'rec_{name}'] = recall_score(targets_np[:, i], preds[:, i], zero_division=0)
        per_class[f'f1_{name}'] = f1_score(targets_np[:, i], preds[:, i], zero_division=0)

    # Macro and micro averages
    macro_prec = precision_score(targets_np, preds, average='macro', zero_division=0)
    macro_rec = recall_score(targets_np, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(targets_np, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(targets_np, preds, average='micro', zero_division=0)

    # mAP
    try:
        mAP = average_precision_score(targets_np, probs, average='macro')
    except ValueError:
        mAP = 0.0

    metrics = {
        'subset_acc': subset_acc,
        'hamming_loss': ham_loss,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'mAP': mAP,
    }
    metrics.update(per_class)
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        all_outputs.append(outputs)
        all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_targets))
    metrics['loss'] = avg_loss
    return metrics


def build_model(model_name: str, num_classes: int, pretrained: bool):
    """Create a timm model with num_classes output.

    Uses cached pretrained weights offline to avoid HF network timeouts.
    Falls back to torchvision pretrained weights when HF cache is unavailable.
    """
    if pretrained:
        # Try timm pretrained (offline cache from HF_HUB_OFFLINE set at module level)
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f'Loaded timm pretrained weights for {model_name}')
            return model
        except Exception:
            pass
        # Fallback to torchvision for supported models
        if model_name in ('resnet18', 'resnet34'):
            print(f'Falling back to torchvision pretrained weights for {model_name}...')
            return _build_tv_pretrained(model_name, num_classes)
        else:
            print(f'Cannot load pretrained weights for {model_name}, training from scratch')
            return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    else:
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _build_tv_pretrained(model_name: str, num_classes: int):
    """Build timm model with torchvision pretrained backbone weights."""
    import torchvision.models as tv_models
    tv_model = getattr(tv_models, model_name)(weights='IMAGENET1K_V1')
    # Replace FC with our num_classes
    in_features = tv_model.fc.in_features
    tv_model.fc = nn.Linear(in_features, num_classes)
    # Create timm model and copy weights
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(tv_model.state_dict())
    print(f'Loaded torchvision pretrained weights for {model_name}')
    return model


def build_optimizer(model, args):
    """Create optimizer with optional differential LR for head vs backbone."""
    if args.opt == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.opt == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f'Unknown optimizer: {args.opt}')


def build_scheduler(optimizer, args):
    """Cosine annealing scheduler with linear warmup (epoch-based)."""
    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch):
        # epoch is 0-indexed LambdaLR step counter
        e = epoch + 1  # convert to 1-indexed epoch number
        if e <= warmup_epochs:
            return max(e, 1) / max(1, warmup_epochs)
        progress = (e - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ILS Interference Multi-label Classification')
    # Data
    parser.add_argument('--data-dir', type=str, default='my_dataset/dataset',
                        help='Root dataset directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        help='timm model name (resnet18, sigresnet18, resnet34, sigresnet34, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--num-classes', type=int, default=5)
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    # Loss
    parser.add_argument('--arc-weight', type=float, default=1.0,
                        help='Extra weight multiplier for Arc class in BCE loss')
    parser.add_argument('--asl', action='store_true', default=False,
                        help='Use Asymmetric Loss (focuses on hard positives)')
    parser.add_argument('--specaugment', action='store_true', default=False,
                        help='Apply time/frequency masking augmentation')
    # Output
    parser.add_argument('--output-dir', type=str, default='my_training_run/ils_resnet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Datasets
    train_dataset = ILSInterferenceDataset(
        args.data_dir, split='train',
        transform=get_transforms(args.img_size, is_train=True, specaugment=args.specaugment),
    )
    val_dataset = ILSInterferenceDataset(
        args.data_dir, split='val',
        transform=get_transforms(args.img_size, is_train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=ILSInterferenceDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=ILSInterferenceDataset.collate_fn,
    )

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Model
    model = build_model(args.model, args.num_classes, args.pretrained)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model} | Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M')

    # Loss
    if args.asl:
        from timm.loss import AsymmetricLossMultiLabel
        criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=1, clip=0.05)
        print('Loss: AsymmetricLossMultiLabel (gamma_neg=4, gamma_pos=1)')
    else:
        # Per-class pos_weight: Arc gets extra weight to address its difficulty
        base_pos_weight = np.array([6660/5940] * 5)  # neg/pos ratio for balanced classes
        base_pos_weight[0] *= args.arc_weight  # Arc (index 0) gets multiplier
        pos_weight = torch.tensor(base_pos_weight, dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if args.arc_weight != 1.0:
            print(f'Loss: BCEWithLogitsLoss with Arc weight x{args.arc_weight}')
        else:
            print('Loss: BCEWithLogitsLoss (balanced)')

    # Optimizer & scheduler
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    except ImportError:
        print('TensorBoard not available, skipping logging')

    # Save config
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    best_mAP = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            for k, v in val_metrics.items():
                if k != 'loss':
                    writer.add_scalar(f'Metrics/{k}', v, epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # Print summary
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_metrics["loss"]:.4f} | '
              f'mAP: {val_metrics["mAP"]:.4f} | Subset Acc: {val_metrics["subset_acc"]:.4f} | '
              f'Macro F1: {val_metrics["macro_f1"]:.4f} | Micro F1: {val_metrics["micro_f1"]:.4f} | '
              f'LR: {current_lr:.2e}')

        # Checkpoint
        is_best = val_metrics['mAP'] > best_mAP
        if is_best:
            best_mAP = val_metrics['mAP']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, output_dir / 'best_model.pth')
            print(f'  -> Best model saved (mAP={best_mAP:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch} (best mAP={best_mAP:.4f} at epoch {best_epoch})')
            break

    if writer is not None:
        writer.close()
    print(f'\nTraining complete. Best mAP: {best_mAP:.4f} at epoch {best_epoch}')

    # ── Final test evaluation ────────────────────────────────────────────────
    print('\n=== Test Evaluation ===')
    test_dataset = ILSInterferenceDataset(
        args.data_dir, split='test',
        transform=get_transforms(args.img_size, is_train=False),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=ILSInterferenceDataset.collate_fn,
    )

    # Load best checkpoint
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device)

    print(f'\nTest Results:')
    print(f'  mAP:          {test_metrics["mAP"]:.4f}')
    print(f'  Subset Acc:   {test_metrics["subset_acc"]:.4f}')
    print(f'  Hamming Loss: {test_metrics["hamming_loss"]:.4f}')
    print(f'  Macro F1:     {test_metrics["macro_f1"]:.4f}')
    print(f'  Micro F1:     {test_metrics["micro_f1"]:.4f}')
    print(f'  Macro Prec:   {test_metrics["macro_precision"]:.4f}')
    print(f'  Macro Rec:    {test_metrics["macro_recall"]:.4f}')
    print()
    for name in LABEL_NAMES:
        print(f'  {name}: Acc={test_metrics[f"acc_{name}"]:.4f} '
              f'Prec={test_metrics[f"prec_{name}"]:.4f} '
              f'Rec={test_metrics[f"rec_{name}"]:.4f} '
              f'F1={test_metrics[f"f1_{name}"]:.4f}')

    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)


if __name__ == '__main__':
    main()
