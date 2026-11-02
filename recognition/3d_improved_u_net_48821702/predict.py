"""Example usage of the trained models: inference, printed results and failure-case figures.

Loads the trained 2D U-Net and/or 3D Improved U-Net, segments selected test cases, prints
per-case metrics, and saves a multi-panel figure per case comparing the two models:
  rows:    sagittal (anterior-posterior x superior-inferior: rectal and bladder interfaces)
           coronal  (left-right x superior-inferior: the 2D model's own input plane)
  columns: input | ground truth | 2D prediction | 3D prediction | 2D errors | 3D errors
           (errors: false positives red, false negatives cyan, true prostate outline green)

Example:
    python predict.py --checkpoint_2d /path/to/unet2d/best.pt \
        --checkpoint_3d /path/to/improved_unet3d/best.pt \
        --data_dir /path/to/HipMRI_Study_open --out_dir /path/to/figures/cases
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')  # write figures to files; no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

import dataset
import utils
from evaluation import load_label_volume, predict_case_2d, predict_case_3d
from train import MODES, build_model

DEFAULT_CASES = ['P040_W0', 'P042_W0', 'P040_W1', 'P040_W3', 'P041_W0']
LABELS = {'unet2d': '2D U-Net', 'improved_unet3d': '3D Improved U-Net'}
SHOWN_CLASSES = {utils.BLADDER: ('bladder', 'gold'),
                 utils.RECTUM: ('rectum', 'tab:brown'),
                 dataset.PROSTATE: ('prostate', 'tab:red')}
CASE_RE = re.compile(r'^P(\d+)_W(\d+)$')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint_2d', default=None, help='best.pt of the 2D U-Net')
    parser.add_argument('--checkpoint_3d', default=None, help='best.pt of the 3D Improved U-Net')
    parser.add_argument('--data_dir', required=True, help='path to HipMRI_Study_open')
    parser.add_argument('--out_dir', required=True, help='where figures are saved')
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES,
                        help="case names such as P040_W0, or 'all' for every test case")
    parser.add_argument('--no_amp', action='store_true', help='disable bfloat16 mixed precision')
    args = parser.parse_args()
    if not (args.checkpoint_2d or args.checkpoint_3d):
        parser.error('give at least one of --checkpoint_2d / --checkpoint_3d')
    return args


def load_model(checkpoint_path, device):
    """Rebuild a trained model from a train.py checkpoint. Returns (model, name, mode)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    name = ckpt['args']['model']
    model = build_model(name, ckpt['args']['base_channels']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded {LABELS[name]} from epoch {ckpt["epoch"] + 1} '
          f'({utils.count_parameters(model):,} parameters)', flush=True)
    return model, name, MODES[name]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def view_slice(volume, view, index):
    """2D display slice with rows = superior-inferior (superior at the top when origin='lower').

    sagittal: volume[index, :, :] -> columns run anterior -> posterior
    coronal:  volume[:, index, :] -> columns run right -> left (radiological convention)
    """
    plane = volume[index, :, :] if view == 'sagittal' else volume[:, index, :]
    return plane.T


def crop_box(masks, margin=20):
    """Row/column slices bounding the union of 2D masks, padded by `margin` pixels."""
    rows, cols = np.where(np.logical_or.reduce(masks))
    shape = masks[0].shape
    return (slice(max(rows.min() - margin, 0), min(rows.max() + margin + 1, shape[0])),
            slice(max(cols.min() - margin, 0), min(cols.max() + margin + 1, shape[1])))


def label_overlay(labels):
    """RGBA image colouring the bladder, rectum and prostate; everything else transparent."""
    rgba = np.zeros(labels.shape + (4,))
    for c, (_, colour) in SHOWN_CLASSES.items():
        rgba[labels == c] = (*to_rgba(colour)[:3], 0.5)
    return rgba


def error_overlay(pred, gt):
    """RGBA image: prostate false positives red, false negatives cyan."""
    p, g = pred == dataset.PROSTATE, gt == dataset.PROSTATE
    rgba = np.zeros(pred.shape + (4,))
    rgba[p & ~g] = (1.0, 0.0, 0.0, 0.8)
    rgba[g & ~p] = (0.0, 1.0, 1.0, 0.8)
    return rgba


def case_figure(case, image, gt, preds, metrics, spacing, path):
    """Multi-panel comparison figure for one case (all volumes are slab-shaped (D, 60, W))."""
    centre = np.argwhere(gt == dataset.PROSTATE).mean(axis=0).round().astype(int)
    views = [('sagittal', centre[0], spacing[1], 'anterior → posterior'),
             ('coronal', centre[1], spacing[0], 'right → left')]
    names = list(preds)
    columns = (['Input', 'Ground truth'] + [LABELS[n] for n in names]
               + [f'{LABELS[n]} errors' for n in names])

    fig, axes = plt.subplots(len(views), len(columns), figsize=(3.2 * len(columns), 7), squeeze=False)
    for row, (view, index, col_spacing, direction) in enumerate(views):
        img, g = view_slice(image, view, index), view_slice(gt, view, index)
        ps = {n: view_slice(preds[n], view, index) for n in names}
        box = crop_box([g == dataset.PROSTATE] + [p == dataset.PROSTATE for p in ps.values()])
        aspect = spacing[2] / col_spacing  # true pixel proportions (rows are slices along S-I)

        panels = ([None, label_overlay(g[box])]
                  + [label_overlay(ps[n][box]) for n in names]
                  + [error_overlay(ps[n][box], g[box]) for n in names])
        for col, overlay in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img[box], cmap='gray', origin='lower', aspect=aspect)
            if overlay is not None:
                ax.imshow(overlay, origin='lower', aspect=aspect)
            if col >= 2 + len(names):  # error panels: outline the true prostate
                ax.contour((g[box] == dataset.PROSTATE).astype(float), levels=[0.5],
                           colors='lime', linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                title = columns[col]
                if col >= 2:
                    title += f'\nDice {metrics[names[(col - 2) % len(names)]]["dice_prostate"]:.3f}'
                ax.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(f'{view}\n{direction} (↑ superior)', fontsize=9)

    handles = ([Patch(color=colour, alpha=0.6, label=label) for label, colour in SHOWN_CLASSES.values()]
               + [Patch(color='red', label='false positive (prostate)'),
                  Patch(color='cyan', label='false negative (prostate)'),
                  Line2D([], [], color='lime', label='true prostate outline')])
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f'{case}: prostate segmentation, 2D vs 3D context', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_metrics(m):
    return (f'Dice {m["dice_prostate"]:.3f} | HD95 {m["hd95_prostate_mm"]:.1f} mm | '
            f'base {m["base_error_mm"]:+.1f} mm | apex {m["apex_error_mm"]:+.1f} mm | '
            f'missed {m["missed_prostate_ml"]:.2f} mL | extra {m["extra_prostate_ml"]:.2f} mL | '
            f'prostate in bladder {m["prostate_in_bladder_ml"]:.2f} mL | '
            f'rectum in prostate {m["rectum_in_prostate_ml"]:.2f} mL')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported() and not args.no_amp

    models = {}
    for path in (args.checkpoint_2d, args.checkpoint_3d):
        if path:
            model, name, mode = load_model(path, device)
            models[name] = (model, mode)

    test_patients = set(dataset.patient_split(args.data_dir)['test'])
    volumes = {k: v for k, v in dataset.find_volumes(args.data_dir).items() if k[0] in test_patients}
    test_slices = dataset.find_slices(args.data_dir, 'test')
    slab = slice(dataset.SLAB_START, dataset.SLAB_START + dataset.SLAB_SIZE)

    if args.cases == ['all']:
        keys = sorted(volumes)
    else:
        keys = []
        for case in args.cases:
            match = CASE_RE.match(case)
            key = (int(match.group(1)), int(match.group(2))) if match else None
            if key not in volumes:
                raise ValueError(f'{case} is not a test case (expected e.g. P040_W0)')
            keys.append(key)

    for key in keys:
        case = f'P{key[0]:03d}_W{key[1]}'
        gt, spacing = load_label_volume(volumes[key][1])
        gt_slab = gt[:, slab, :]
        preds, metrics, image = {}, {}, None

        for name, (model, mode) in models.items():
            if mode == '2d':
                pred_slab, image_slab, _, seconds = predict_case_2d(model, test_slices[key], device, use_amp)
            else:
                pred_full, image_full, seconds = predict_case_3d(model, volumes[key][0], device, use_amp)
                pred_slab, image_slab = pred_full[:, slab, :], image_full[:, slab, :]
                image = image_slab  # prefer the volume-normalised image for display
            if image is None:
                image = image_slab
            preds[name] = pred_slab
            metrics[name], _ = utils.volume_metrics(pred_slab, gt_slab, spacing)
            print(f'{case} | {LABELS[name]:17s} | {format_metrics(metrics[name])} | '
                  f'{seconds * 1000:.0f} ms', flush=True)

        path = os.path.join(args.out_dir, f'{case}.png')
        case_figure(case, image, gt_slab, preds, metrics, spacing, path)
        print(f'  saved {path}', flush=True)


if __name__ == '__main__':
    main()