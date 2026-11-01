"""Evaluate a trained model on held-out HipMRI volumes with volume-level metrics.

2D models are run on each case's 60 coronal slices and their predictions are restacked
into the 3D slab (axis-1 indices SLAB_START .. SLAB_START + SLAB_SIZE - 1), so 2D and 3D
models are scored against the same 3D ground truth with the same metrics.

Example:
    python evaluation.py --checkpoint /path/to/unet2d/best.pt \
        --data_dir /path/to/HipMRI_Study_open --out_dir /path/to/eval/unet2d
"""
import argparse
import csv
import json
import os
import time

import nibabel as nib
import numpy as np
import torch

import dataset
import utils
from train import MODES, build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='path to a best.pt from train.py')
    parser.add_argument('--data_dir', required=True, help='path to HipMRI_Study_open')
    parser.add_argument('--out_dir', required=True, help='where metrics and predictions are saved')
    parser.add_argument('--split', default='test', choices=dataset.SPLITS)
    parser.add_argument('--no_amp', action='store_true', help='disable bfloat16 mixed precision')
    parser.add_argument('--no_save_predictions', action='store_true')
    return parser.parse_args()


def restack(slices):
    """(SLAB_SIZE, H, W) coronal slices -> (H, SLAB_SIZE, W) slab of the 3D volume.

    Inverse of the dataset mapping  2D slice k == flipud(volume[:, SLAB_START + k, :]).
    """
    return np.stack([np.flipud(s) for s in slices], axis=1)


@torch.no_grad()
def predict_case_2d(model, case_slices, device, use_amp):
    """Segment one case's 2D slices and restack everything into the 3D slab.

    Returns (pred_slab, image_slab, gt_slab_from_2d_masks, inference_seconds).
    """
    if [k for k, _, _ in case_slices] != list(range(dataset.SLAB_SIZE)):
        raise ValueError('Expected slices 0..SLAB_SIZE-1 for every case')
    images = dataset.load_data_2D([img for _, img, _ in case_slices], normImage=True)
    masks = dataset.load_data_2D([msk for _, _, msk in case_slices], dtype=np.uint8)

    x = torch.from_numpy(images).unsqueeze(1).to(device)  # (60, 1, H, W)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = model(x).argmax(dim=1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.time() - start

    pred = pred.cpu().numpy().astype(np.uint8)
    return restack(pred), restack(images), restack(masks), seconds

@torch.no_grad()
def predict_case_3d(model, image_path, device, use_amp):
    """Segment one whole 3D volume.

    Returns (pred_volume, image_volume, inference_seconds), each volume (D, H, W).
    """
    image = dataset.load_data_3D([image_path], normImage=True)[0]  # z-scored as in training
    x = torch.from_numpy(image)[None, None].to(device)  # (1, 1, D, H, W)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = model(x).argmax(dim=1)[0]
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.time() - start
    return pred.cpu().numpy().astype(np.uint8), image, seconds


def load_label_volume(path):
    """3D label volume as uint8, plus its voxel spacing (mm) from the NIfTI header."""
    nifti = nib.load(path)
    labels = nifti.get_fdata(caching='unchanged')
    while labels.ndim > 3:
        labels = labels[..., 0]
    spacing = tuple(float(s) for s in nifti.header.get_zooms()[:3])
    return np.rint(labels).astype(np.uint8), spacing


def summarise(rows):
    """Mean, standard deviation and median of every numeric metric across cases."""
    summary = {}
    for key in rows[0]:
        if key == 'case':
            continue
        values = np.array([r[key] for r in rows], dtype=float)
        summary[key] = {'mean': float(np.nanmean(values)), 'std': float(np.nanstd(values)),
                        'median': float(np.nanmedian(values)),
                        'n_nan': int(np.isnan(values).sum())}
    return summary


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported() and not args.no_amp

    # Rebuild the model exactly as it was trained
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt['args']
    name, mode = train_args['model'], MODES[train_args['model']]
    model = build_model(name, train_args['base_channels']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'{name} from epoch {ckpt["epoch"] + 1} | {utils.count_parameters(model):,} parameters',
          flush=True)

    # Cases in the requested split
    patients = set(dataset.patient_split(args.data_dir)[args.split])
    volumes = {k: v for k, v in dataset.find_volumes(args.data_dir).items() if k[0] in patients}
    case_slices = dataset.find_slices(args.data_dir, args.split) if mode == '2d' else None
    slab = slice(dataset.SLAB_START, dataset.SLAB_START + dataset.SLAB_SIZE)

    # Warm-up pass so the first case's timing excludes one-off GPU start-up costs
    warmup_shape = dataset.EXPECTED_SHAPE_2D if mode == '2d' else dataset.EXPECTED_SHAPE_3D
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        model(torch.zeros(1, 1, *warmup_shape, device=device))

    rows, profile_rows = [], []
    for key in sorted(volumes):
        case = f'P{key[0]:03d}_W{key[1]}'
        gt, spacing = load_label_volume(volumes[key][1])
        gt_slab = gt[:, slab, :]

        full_metrics, pred_full = {}, None
        if mode == '2d':
            pred_slab, image_slab, gt_from_2d, seconds = predict_case_2d(
                model, case_slices[key], device, use_amp)
            # Re-verify the 2D <-> 3D mapping on every evaluated case
            if not np.array_equal(gt_from_2d, gt_slab):
                raise RuntimeError(f'Restacked 2D masks do not match the 3D labels for {case}')
        else:
            pred_full, image_full, seconds = predict_case_3d(model, volumes[key][0], device, use_amp)
            # Primary comparison: the same slab the 2D model covers
            pred_slab, image_slab = pred_full[:, slab, :], image_full[:, slab, :]
            # Secondary: the whole volume, which only the 3D model covers
            full, _ = utils.volume_metrics(pred_full, gt, spacing)
            full_metrics = {f'full_{k}': v for k, v in full.items()}

        metrics, (z, slice_dice, relative) = utils.volume_metrics(pred_slab, gt_slab, spacing)
        rows.append({'case': case, **metrics, **full_metrics, 'inference_s': seconds})
        profile_rows += [{'case': case, 'slice': int(i), 'relative_position': float(r), 'dice': float(d)}
                         for i, d, r in zip(z, slice_dice, relative)]
        print(f'{case} | prostate Dice {metrics["dice_prostate"]:.3f} | '
              f'HD95 {metrics["hd95_prostate_mm"]:.1f} mm | base {metrics["base_error_mm"]:+.1f} mm | '
              f'apex {metrics["apex_error_mm"]:+.1f} mm | components {metrics["prostate_components"]} | '
              f'in rectum {metrics["prostate_in_rectum_ml"]:.2f} mL | {seconds * 1000:.0f} ms', flush=True)

        if not args.no_save_predictions:
            os.makedirs(os.path.join(args.out_dir, 'predictions'), exist_ok=True)
            extra = {'pred_full': pred_full} if pred_full is not None else {}
            np.savez_compressed(os.path.join(args.out_dir, 'predictions', f'{case}.npz'),
                                pred=pred_slab, gt=gt_slab, image=image_slab.astype(np.float16),
                                spacing=np.array(spacing), **extra)

    # Save per-case metrics, the slice profile and a summary
    write_csv(os.path.join(args.out_dir, 'per_volume.csv'), rows)
    write_csv(os.path.join(args.out_dir, 'slice_profile.csv'), profile_rows)
    summary = summarise(rows)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({'model': name, 'checkpoint': args.checkpoint, 'split': args.split,
                   'n_cases': len(rows), 'metrics': summary}, f, indent=1)

    print('\nSummary over', len(rows), 'cases (mean ± std, median):')
    for key, s in summary.items():
        print(f'  {key:24s} {s["mean"]:8.3f} ± {s["std"]:.3f}   median {s["median"]:.3f}'
              + (f'   ({s["n_nan"]} NaN)' if s['n_nan'] else ''))


if __name__ == '__main__':
    main()