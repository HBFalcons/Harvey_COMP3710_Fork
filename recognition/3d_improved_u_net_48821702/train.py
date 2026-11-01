# Contains code that trains the models under the training data

"""Train, validate and test segmentation models on the HipMRI dataset.

Examples:
    Colab:
        python train.py --model unet2d --data_dir /content/data/HipMRI_Study_open \
            --out_dir /content/drive/MyDrive/comp3710/checkpoints/unet2d
    Rangpur:
        python train.py --model unet2d --data_dir /home/groups/comp3710/HipMRI_Study_open \
            --out_dir ./outputs/unet2d
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use('Agg')  # write plots to files; no display needed (e.g. in Slurm jobs)
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset
import modules
import utils

MODES = {'unet2d': '2d', 'improved_unet3d': '3d'}  # model name -> data mode

# Per-model defaults, used when the corresponding flag is not given on the command line
MODEL_DEFAULTS = {
    'unet2d':          {'base_channels': 64, 'lr': 1e-3, 'batch_size': 16, 'epochs': 50},
    'improved_unet3d': {'base_channels': 16, 'lr': 5e-4, 'batch_size': 2,  'epochs': 150},
}

SPACING = {'2d': dataset.SPACING_2D, '3d': dataset.SPACING_3D}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(MODES), default='unet2d')
    parser.add_argument('--data_dir', required=True, help='path to HipMRI_Study_open')
    parser.add_argument('--out_dir', required=True, help='where checkpoints, logs and plots are saved')
    parser.add_argument('--epochs', type=int, default=None, help='default depends on --model')
    parser.add_argument('--batch_size', type=int, default=None, help='default depends on --model')
    parser.add_argument('--lr', type=float, default=None, help='default depends on --model')
    parser.add_argument('--base_channels', type=int, default=None, help='default depends on --model')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_amp', action='store_true', help='disable bfloat16 mixed precision')
    parser.add_argument('--no_augment', action='store_true', help='disable training augmentation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='continue from out_dir/last.pt')

    args = parser.parse_args()
    for key, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def set_seed(seed):
    """Seed every random number generator for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, base_channels):
    if name == 'unet2d':
        return modules.UNet2D(in_channels=1, num_classes=dataset.NUM_CLASSES,
                              base_channels=base_channels)
    if name == 'improved_unet3d':
        return modules.ImprovedUNet3D(in_channels=1, num_classes=dataset.NUM_CLASSES,
                                      base_channels=base_channels)
    raise ValueError(f'Unknown model {name!r}')


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, spacing, augment):
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total_loss, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if augment:
            images, labels = dataset.augment_batch(images, labels, spacing)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits, labels)  # DiceCELoss computes in float32 internally
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        seen += len(images)
    return total_loss / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """Mean loss and per-class Dice over a whole split (Dice pooled over all its voxels)."""
    model.eval()
    total_loss, seen, stats = 0.0, 0, None
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        total_loss += criterion(logits, labels).item() * len(images)
        seen += len(images)
        batch_stats = utils.dice_stats(logits.argmax(dim=1), labels, dataset.NUM_CLASSES)
        stats = batch_stats if stats is None else [a + b for a, b in zip(stats, batch_stats)]
    return total_loss / seen, utils.dice_from_stats(*stats)


def plot_history(history, path):
    """Save training/validation loss and per-class validation Dice curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history['train_loss'], label='train')
    ax_loss.plot(epochs, history['val_loss'], label='validation')
    ax_loss.set(xlabel='epoch', ylabel='Dice + CE loss', title='Loss')
    ax_loss.legend()

    dice = np.array(history['val_dice'])  # (epochs, classes)
    for c in range(1, dataset.NUM_CLASSES):
        ax_dice.plot(epochs, dice[:, c], label=dataset.CLASS_NAMES[c],
                     linewidth=2.5 if c == dataset.PROSTATE else 1)
    ax_dice.set(xlabel='epoch', ylabel='Dice', title='Validation Dice per class', ylim=(0, 1))
    ax_dice.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def format_dice(dice):
    return ' '.join(f'{name[:4]} {d:.3f}' for name, d in zip(dataset.CLASS_NAMES[1:], dice[1:]))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported() and not args.no_amp
    mode = MODES[args.model]

    # Data
    train_loader = dataset.get_dataloader(args.data_dir, 'train', mode, args.batch_size, args.num_workers)
    val_loader = dataset.get_dataloader(args.data_dir, 'validate', mode, args.batch_size, args.num_workers)

    # Model, loss, optimiser, learning-rate schedule
    model = build_model(args.model, args.base_channels).to(device)
    criterion = modules.DiceCELoss(num_classes=dataset.NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'lr': [], 'epoch_time': []}
    start_epoch, best_dice = 0, -1.0
    last_path = os.path.join(args.out_dir, 'last.pt')
    best_path = os.path.join(args.out_dir, 'best.pt')

    if args.resume and os.path.exists(last_path):
        # weights_only=False: the checkpoint also stores plain-Python training state (history, args)
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        history, best_dice, start_epoch = ckpt['history'], ckpt['best_dice'], ckpt['epoch'] + 1
        print(f'Resumed from {last_path} at epoch {start_epoch + 1}', flush=True)

    print(f'{args.model}: {utils.count_parameters(model):,} parameters | device {device} | '
          f'bf16 AMP {use_amp} | augmentation {not args.no_augment}', flush=True)
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp,
                                     SPACING[mode], augment=not args.no_augment)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, use_amp)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['epoch_time'].append(time.time() - start)

        # Keep the epoch with the best validation prostate Dice (the primary target)
        if val_dice[dataset.PROSTATE] > best_dice:
            best_dice = val_dice[dataset.PROSTATE]
            torch.save({'model': model.state_dict(), 'args': vars(args),
                        'epoch': epoch, 'val_dice': val_dice}, best_path)

        # Save everything needed to resume after a disconnect
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'history': history,
                    'best_dice': best_dice, 'epoch': epoch, 'args': vars(args)}, last_path)
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        plot_history(history, os.path.join(args.out_dir, 'training_curves.png'))

        print(f'epoch {epoch + 1:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | '
              f'{format_dice(val_dice)} | {history["epoch_time"][-1]:.0f}s', flush=True)

    # Test the best checkpoint on the held-out patients
    del train_loader  # free the pre-loaded training split before loading the test split
    test_loader = dataset.get_dataloader(args.data_dir, 'test', mode, args.batch_size, args.num_workers)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    test_loss, test_dice = evaluate(model, test_loader, criterion, device, use_amp)

    results = {
        'model': args.model,
        'best_epoch': best['epoch'] + 1,
        'parameters': utils.count_parameters(model),
        'peak_train_vram_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
        'test_loss': test_loss,
        'test_dice': dict(zip(dataset.CLASS_NAMES, test_dice)),
        'args': vars(args),
    }
    with open(os.path.join(args.out_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nTest (best epoch {results["best_epoch"]}) | loss {test_loss:.4f} | {format_dice(test_dice)}',
          flush=True)


if __name__ == '__main__':
    main()