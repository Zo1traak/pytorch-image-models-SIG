"""Hard negative mining fine-tuning for Arc precision improvement.

Loads best ArcW3+SpecAug model, freezes backbone layers, and fine-tunes
classifier with dynamic Arc-negative weighting to suppress false positives.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_ils import ILSInterferenceDataset, get_transforms, ArcAwareSpecAugment, LABEL_NAMES
from train_ils import compute_metrics  # reuse metrics


def build_model_from_checkpoint(ckpt_path: str, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']
    model = timm.create_model(args['model'], pretrained=False, num_classes=5)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model, args


def freeze_backbone(model):
    """Freeze all layers except the final classifier (fc)."""
    for name, param in model.named_parameters():
        if 'fc' not in name:  # timm ResNet uses 'fc' for the classifier
            param.requires_grad = False
    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')
    return model


def fine_tune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load best model
    model, orig_args = build_model_from_checkpoint(args.resume, device)
    model = freeze_backbone(model)

    # Data
    specaug = ArcAwareSpecAugment(mode='current') if args.specaug else None
    train_ds = ILSInterferenceDataset(
        args.data_dir, split='train',
        transform=get_transforms(args.img_size, is_train=True),
        specaug=specaug,
    )
    val_ds = ILSInterferenceDataset(
        args.data_dir, split='val',
        transform=get_transforms(args.img_size, is_train=False),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=ILSInterferenceDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=ILSInterferenceDataset.collate_fn)

    # Loss: ArcW3 for main BCE, plus extra Arc_neg focus
    base_pos_weight = np.array([6660/5940] * 5)
    base_pos_weight[0] *= 3.0  # ArcW3
    pos_weight = torch.tensor(base_pos_weight, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mAP = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Per-sample loss (reduction='none')
            loss_per_sample = criterion(outputs, targets)  # [B, 5]
            # Extra weight for Arc dimension on hard negatives
            with torch.no_grad():
                probs = torch.sigmoid(outputs[:, 0])
                y_arc = targets[:, 0]
                # Hard negative: y_Arc=0 but p_Arc > 0.5
                hard_neg_mask = (y_arc == 0) & (probs > 0.5)
                arc_weights = torch.ones_like(y_arc)
                arc_weights[hard_neg_mask] = args.hard_neg_weight  # extra penalty

            # Weighted sum: normal weight for all, extra for Arc on hard negatives
            weighted_loss = loss_per_sample.clone()
            weighted_loss[:, 0] *= arc_weights  # Scale Arc dimension

            loss = weighted_loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        all_out, all_tgt = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets).mean()
                val_loss += loss.item() * images.size(0)
                all_out.append(outputs)
                all_tgt.append(targets)

        val_loss /= len(val_ds)
        metrics = compute_metrics(torch.cat(all_out), torch.cat(all_tgt))
        metrics['loss'] = val_loss

        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:2d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'mAP: {metrics["mAP"]:.4f} | Arc F1: {metrics["f1_Arc"]:.4f} | '
              f'Arc Prec: {metrics["prec_Arc"]:.4f} | Arc Rec: {metrics["rec_Arc"]:.4f} | '
              f'Macro F1: {metrics["macro_f1"]:.4f} | LR: {lr:.2e}')

        if metrics['mAP'] > best_mAP:
            best_mAP = metrics['mAP']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': metrics,
            }, args.output_dir / 'best_model.pth')
            print(f'  -> Best model (mAP={best_mAP:.4f}, Arc F1={metrics["f1_Arc"]:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Test evaluation
    print('\n=== Test Evaluation ===')
    test_ds = ILSInterferenceDataset(
        args.data_dir, split='test',
        transform=get_transforms(args.img_size, is_train=False),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             collate_fn=ILSInterferenceDataset.collate_fn)

    ckpt = torch.load(args.output_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_out, all_tgt = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            all_out.append(outputs)
            all_tgt.append(targets)

    test_metrics = compute_metrics(torch.cat(all_out), torch.cat(all_tgt))
    print(f'  mAP: {test_metrics["mAP"]:.4f}')
    print(f'  Macro F1: {test_metrics["macro_f1"]:.4f}')
    for name in LABEL_NAMES:
        print(f'  {name}: F1={test_metrics[f"f1_{name}"]:.4f} '
              f'Prec={test_metrics[f"prec_{name}"]:.4f} '
              f'Rec={test_metrics[f"rec_{name}"]:.4f}')

    with open(args.output_dir / 'test_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arc Hard Negative Fine-tuning')
    parser.add_argument('--data-dir', type=str, default='my_dataset/dataset')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4, help='Lower LR for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hard-neg-weight', type=float, default=3.0,
                        help='Extra Arc BCE weight for hard negatives (y=0, p>0.5)')
    parser.add_argument('--resume', type=str,
                        default='my_training_run/sigresnet18_arcw3_specaug/best_model.pth')
    parser.add_argument('--specaug', action='store_true', default=True)
    parser.add_argument('--output-dir', type=str,
                        default='my_training_run/finetune_hardneg')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fine_tune(args)
