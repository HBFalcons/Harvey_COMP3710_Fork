"""Profile the computational cost of the 2D U-Net and 3D Improved U-Net under identical conditions.

Measures, on the same GPU with bfloat16 mixed precision:
  - parameter count and checkpoint size
  - peak GPU memory and step time for one training step at each model's training batch size
  - peak GPU memory and latency to segment one whole scan (2D: all 60 slices; 3D: one volume)
  - actual epoch and total training time, read from each run's history.json

Synthetic inputs of the real shapes are used: memory and speed depend on tensor shape, not
image content, so no data loading is needed.

Example:
    python profiling.py --ckpt_root /path/to/checkpoints --out_dir /path/to/results/profiling
"""
import argparse
import json
import os
import statistics
import time

import torch

import dataset
import modules
import utils
from train import MODEL_DEFAULTS, MODES, build_model

INPUT_SHAPES = {'2d': dataset.EXPECTED_SHAPE_2D, '3d': dataset.EXPECTED_SHAPE_3D}
INPUTS_PER_SCAN = {'2d': dataset.SLAB_SIZE, '3d': 1}  # model inputs needed to segment one scan
LABELS = {'unet2d': '2D U-Net', 'improved_unet3d': '3D Improved U-Net'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt_root', default=None,
                        help='folder containing <model>/history.json and <model>/best.pt (optional)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--models', nargs='+', default=['unet2d', 'improved_unet3d'],
                        choices=sorted(MODES))
    parser.add_argument('--repeats', type=int, default=20, help='timed runs per measurement')
    parser.add_argument('--warmup', type=int, default=5, help='untimed runs before timing')
    return parser.parse_args()


def timed(fn, repeats, warmup):
    """Mean and standard deviation (seconds) of fn() over `repeats` runs, after `warmup` runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.mean(times), statistics.stdev(times)


def peak_gb():
    return (torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9)


def profile_model(name, repeats, warmup, device):
    mode, cfg = MODES[name], MODEL_DEFAULTS[name]
    shape = INPUT_SHAPES[mode]
    torch.manual_seed(0)
    model = build_model(name, cfg['base_channels']).to(device)
    criterion = modules.DiceCELoss(num_classes=dataset.NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    result = {'model': name, 'parameters': utils.count_parameters(model),
              'train_batch_size': cfg['batch_size'], 'inputs_per_scan': INPUTS_PER_SCAN[mode]}

    # --- One training step (forward, backward, Adam update) at the training batch size ---
    x = torch.randn(cfg['batch_size'], 1, *shape, device=device)
    y = torch.randint(0, dataset.NUM_CLASSES, (cfg['batch_size'], *shape), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        criterion(logits, y).backward()
        optimizer.step()

    model.train()
    train_step()  # creates the Adam state, so it is included in the measured memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result['train_step_s'] = timed(train_step, repeats, warmup)
    result['train_peak_allocated_gb'], result['train_peak_reserved_gb'] = peak_gb()

    # --- Inference on one whole scan ---
    del x, y, optimizer
    model.zero_grad(set_to_none=True)
    model.eval()
    scan = torch.randn(INPUTS_PER_SCAN[mode], 1, *shape, device=device)

    @torch.no_grad()
    def infer():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(scan).argmax(dim=1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result['inference_per_scan_s'] = timed(infer, repeats, warmup)
    result['inference_peak_allocated_gb'], result['inference_peak_reserved_gb'] = peak_gb()

    del model, scan
    torch.cuda.empty_cache()
    return result


def add_training_history(result, ckpt_root):
    """Actual training time and checkpoint size from the real training run, if available."""
    run_dir = os.path.join(ckpt_root, result['model'])
    history_path = os.path.join(run_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            epoch_times = json.load(f)['epoch_time']
        result['epochs'] = len(epoch_times)
        result['epoch_time_s'] = (statistics.mean(epoch_times), statistics.stdev(epoch_times))
        result['total_training_min'] = sum(epoch_times) / 60
    best_path = os.path.join(run_dir, 'best.pt')
    if os.path.exists(best_path):
        result['checkpoint_mb'] = os.path.getsize(best_path) / 1e6


def markdown_table(results):
    """Format the results as a GitHub Markdown table for the README."""
    def ms(pair):
        return f'{pair[0] * 1000:.1f} ± {pair[1] * 1000:.1f}'

    rows = [
        ('Parameters', lambda r: f'{r["parameters"]:,}'),
        ('Checkpoint size (MB)', lambda r: f'{r["checkpoint_mb"]:.1f}' if 'checkpoint_mb' in r else '-'),
        ('Training batch', lambda r: f'{r["train_batch_size"]} {"slices" if r["inputs_per_scan"] > 1 else "volumes"}'),
        ('Peak training VRAM, allocated / reserved (GB)',
         lambda r: f'{r["train_peak_allocated_gb"]:.2f} / {r["train_peak_reserved_gb"]:.2f}'),
        ('Training step time (ms)', lambda r: ms(r['train_step_s'])),
        ('Epoch time (s)', lambda r: f'{r["epoch_time_s"][0]:.1f}' if 'epoch_time_s' in r else '-'),
        ('Epochs / total training time (min)',
         lambda r: f'{r["epochs"]} / {r["total_training_min"]:.0f}' if 'epochs' in r else '-'),
        ('Peak inference VRAM per scan, allocated / reserved (GB)',
         lambda r: f'{r["inference_peak_allocated_gb"]:.2f} / {r["inference_peak_reserved_gb"]:.2f}'),
        ('Inference latency per scan (ms)', lambda r: ms(r['inference_per_scan_s'])),
    ]
    header = '| Metric | ' + ' | '.join(LABELS.get(r['model'], r['model']) for r in results) + ' |'
    divider = '|---|' + '---|' * len(results)
    body = ['| ' + label + ' | ' + ' | '.join(fmt(r) for r in results) + ' |' for label, fmt in rows]
    return '\n'.join([header, divider] + body)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('Profiling requires a CUDA GPU')
    device = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for name in args.models:
        print(f'Profiling {name} ...', flush=True)
        result = profile_model(name, args.repeats, args.warmup, device)
        if args.ckpt_root:
            add_training_history(result, args.ckpt_root)
        results.append(result)

    info = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'precision': 'bfloat16 autocast', 'repeats': args.repeats, 'warmup': args.warmup}
    with open(os.path.join(args.out_dir, 'profiling.json'), 'w') as f:
        json.dump({'info': info, 'results': results}, f, indent=1)
    table = markdown_table(results)
    with open(os.path.join(args.out_dir, 'profiling_table.md'), 'w') as f:
        f.write(table + '\n')
    print(f'\nGPU: {info["gpu"]} | PyTorch {info["torch"]} | {info["precision"]}\n')
    print(table)


if __name__ == '__main__':
    main()