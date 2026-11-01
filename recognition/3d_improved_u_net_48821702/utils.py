# Contains helper functions for the 3D Improved U-Net and benchmark 2D U-Net architectures

"""Shared helpers: segmentation metrics and model statistics."""
import torch


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