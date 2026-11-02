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