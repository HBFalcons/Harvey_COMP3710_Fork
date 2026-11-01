# Contains helper functions for the 3D Improved U-Net and benchmark 2D U-Net architectures

"""Shared helpers: segmentation metrics and model statistics."""
import torch
import numpy as np
from scipy import ndimage

import dataset


def count_parameters(model):
    """Total number of model params"""
    return sum(p.numel() for p in model.parameters())


def dice_stats(preds, labels, num_classes):
    """Per-class overlap statistics for hard (argmax) predictions.

    preds, labels: (N, *spatial) class-index tensors.
    Returns three (C,) tensors on the CPU: intersection, predicted voxel count and
    true voxel count, each summed over the batch. Summing these across batches gives
    Dice pooled over a whole split.
    """
    preds, labels = preds.flatten(), labels.flatten()
    intersection = torch.bincount(labels[preds == labels], minlength=num_classes)
    pred_count = torch.bincount(preds, minlength=num_classes)
    true_count = torch.bincount(labels, minlength=num_classes)
    return intersection.cpu(), pred_count.cpu(), true_count.cpu()


def dice_from_stats(intersection, pred_count, true_count):
    """Dice per class = 2|P ∩ T| / (|P| + |T|), as a list. NaN for a class absent from both."""
    denominator = (pred_count + true_count).double()
    return (2 * intersection.double() / denominator).tolist()


# ---------------------------------------------------------------------------
# Volume-level evaluation metrics (NumPy / SciPy, used by evaluate.py)
# ---------------------------------------------------------------------------
BLADDER = dataset.CLASS_NAMES.index('bladder')
RECTUM = dataset.CLASS_NAMES.index('rectum')


def dice_iou(pred, gt, num_classes):
    """Per-class Dice and IoU for two label volumes. NaN for a class absent from both."""
    dice, iou = [], []
    for c in range(num_classes):
        p, g = pred == c, gt == c
        intersection = np.logical_and(p, g).sum()
        total = p.sum() + g.sum()
        union = total - intersection
        dice.append(2 * intersection / total if total else np.nan)
        iou.append(intersection / union if union else np.nan)
    return dice, iou


def _surface(mask):
    """Boundary voxels of a binary mask."""
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred, gt, spacing):
    """95th-percentile symmetric Hausdorff distance (mm) between two binary masks.

    Distances are measured between the two surfaces in millimetres using the volume's
    own voxel spacing. NaN if either mask is empty.
    """
    if not pred.any() or not gt.any():
        return np.nan
    pred_surface, gt_surface = _surface(pred), _surface(gt)
    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dist_to_gt[pred_surface], dist_to_pred[gt_surface]])
    return float(np.percentile(distances, 95))


def extent_errors(pred, gt, slice_spacing, axis=2):
    """Signed base (superior) and apex (inferior) extent errors in mm along `axis`.

    The index along `axis` increases superiorly (LPS orientation). Positive values mean
    the prediction extends beyond the ground truth (over-contouring); negative values
    mean it stops short (under-contouring). NaN if the prediction is empty.
    """
    other = tuple(a for a in range(pred.ndim) if a != axis)
    pred_z = np.where(pred.any(axis=other))[0]
    gt_z = np.where(gt.any(axis=other))[0]
    if pred_z.size == 0 or gt_z.size == 0:
        return np.nan, np.nan
    base_error = (pred_z.max() - gt_z.max()) * slice_spacing
    apex_error = (gt_z.min() - pred_z.min()) * slice_spacing
    return float(base_error), float(apex_error)


def slice_dice_profile(pred, gt, axis=2):
    """Dice of every slice along `axis` containing the structure in the prediction or ground truth.

    Returns (slice_indices, dice, relative_position), where relative position is 0 at the
    ground truth's most inferior slice (apex) and 1 at its most superior slice (base).
    Slices outside 0..1 contain only false positives (Dice 0 there).
    """
    other = tuple(a for a in range(pred.ndim) if a != axis)
    intersection = np.logical_and(pred, gt).sum(axis=other)
    pred_sum, gt_sum = pred.sum(axis=other), gt.sum(axis=other)
    idx = np.where(pred_sum + gt_sum > 0)[0]
    dice = 2 * intersection[idx] / (pred_sum[idx] + gt_sum[idx])
    gt_z = np.where(gt_sum > 0)[0]
    relative = (idx - gt_z.min()) / max(gt_z.max() - gt_z.min(), 1)
    return idx, dice, relative


def volume_metrics(pred, gt, spacing):
    """All volume-level metrics for one case.

    pred, gt: integer label volumes of the same shape; spacing: voxel size (mm) per axis,
    with axis 2 the superior-inferior axis. Returns (metrics dict, slice Dice profile).
    """
    metrics = {}
    dice, iou = dice_iou(pred, gt, dataset.NUM_CLASSES)
    for name, d, j in zip(dataset.CLASS_NAMES[1:], dice[1:], iou[1:]):
        metrics[f'dice_{name}'] = float(d)
        metrics[f'iou_{name}'] = float(j)

    p, g = pred == dataset.PROSTATE, gt == dataset.PROSTATE
    voxel_ml = float(np.prod(spacing)) / 1000.0
    metrics['hd95_prostate_mm'] = hd95(p, g, spacing)
    metrics['base_error_mm'], metrics['apex_error_mm'] = extent_errors(p, g, spacing[2])
    metrics['prostate_components'] = int(ndimage.label(p)[1])
    metrics['rectum_in_prostate_ml'] = float(((pred == RECTUM) & g).sum() * voxel_ml)
    metrics['prostate_in_rectum_ml'] = float((p & (gt == RECTUM)).sum() * voxel_ml)
    metrics['prostate_in_bladder_ml'] = float((p & (gt == BLADDER)).sum() * voxel_ml)
    metrics['missed_prostate_ml'] = float((g & ~p).sum() * voxel_ml)
    metrics['extra_prostate_ml'] = float((p & ~g).sum() * voxel_ml)
    
    return metrics, slice_dice_profile(p, g)