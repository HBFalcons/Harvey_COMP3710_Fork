# Data loading, splitting, and augmentation for the HipMRI dataset
# Based on dataset, splitting is done at the patient level, so that all slices from a given patient are either in the training or validation set.

import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ['background', 'body', 'bone', 'bladder', 'rectum', 'prostate']
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = 5

# The 2D keras slices are coronal cuts of the 3D volumes (orientation LPS):
#    2D slice k  ==  flipud(volume[:, SLAB_START + k, :])   for k in 0..SLAB_SIZE-1
# Verified by exact correlation matching for all 211 volumes and their masks.
SLAB_START = 98 # 2D slice 0 is 3D slice 98
SLAB_SIZE = 60 # 60 2D slices per 3D volume, coronal cuts of the 3D volume in the LPS orientation

SPLITS = ('train', 'validate', 'test')

# Filename patterns
#   3D image: M004_Week0_LFOV.nii.gz      3D label: M004_Week0_SEMANTIC.nii.gz
#   2D image: case_004_week_0_slice_12.nii.gz   2D mask: seg_004_week_0_slice_12.nii.gz
_VOLUME_RE = re.compile(r'^([A-Z])(\d+)_Week(\d+)_LFOV\.nii\.gz$') #3D file name template
_SLICE_RE = re.compile(r'^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$') #2D file name template


def find_volumes(data_root):
    """Locate every 3D MR volume and its matching label volume.

    Returns:
        dict mapping (patient_number, week) -> (image_path, label_path)
    """
    mr_dir = os.path.join(data_root, 'semantic_MRs')
    label_dir = os.path.join(data_root, 'semantic_labels_only')

    volumes = {}
    for fname in sorted(os.listdir(mr_dir)):
        match = _VOLUME_RE.match(fname)
        if match is None: # Only proceed with files that match the expected 3D volume filename pattern
            continue
        letter, number, week = match.groups()
        label_path = os.path.join(label_dir, f'{letter}{number}_Week{week}_SEMANTIC.nii.gz')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'No label volume found for {fname}')
        volumes[(int(number), int(week))] = (os.path.join(mr_dir, fname), label_path)
    return volumes


def find_slices(data_root, split_name):
    """Locate every 2D slice and its mask in one of the keras_slices_data splits.

    Returns:
        dict mapping (patient_number, week) -> list of (slice_index, image_path, mask_path),
        sorted by slice index so each list can be restacked into a volume in order.
    """
    base = os.path.join(data_root, 'keras_slices_data')
    image_dir = os.path.join(base, f'keras_slices_{split_name}')
    mask_dir = os.path.join(base, f'keras_slices_seg_{split_name}')

    cases = defaultdict(list)
    for fname in os.listdir(image_dir):
        match = _SLICE_RE.match(fname)
        if match is None: # Only proceed with files that match the expected 2D slice filename pattern
            continue
        patient, week, slice_idx = (int(g) for g in match.groups())
        mask_path = os.path.join(mask_dir, 'seg' + fname[len('case'):])
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'No mask found for {fname}')
        cases[(patient, week)].append((slice_idx, os.path.join(image_dir, fname), mask_path))

    for slices in cases.values():
        slices.sort()
    return dict(cases)


def check_no_leakage(split):
    """Raise an error if any patient appears in more than one split."""
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(split[a]) & set(split[b])
            if overlap:
                raise ValueError(f'Patient leakage between {a} and {b}: {sorted(overlap)}')


def patient_split(data_root):
    """Build the patient-level split shared by the 2D baseline and the 3D model.

    The split is taken from the official keras_slices_data train/validate/test folders,
    which are already separated by patient. Using the same patient lists for the 3D
    volumes means both models are evaluated on exactly the same held-out cases.

    Returns:
        dict mapping split name -> sorted list of patient numbers
    """
    split = {s: sorted({patient for patient, _ in find_slices(data_root, s)}) for s in SPLITS}
    check_no_leakage(split)

    # Every 3D patient must be assigned to exactly one split
    all_3d_patients = {patient for patient, _ in find_volumes(data_root)}
    unassigned = all_3d_patients - set().union(*split.values())
    if unassigned:
        raise ValueError(f'3D patients missing from the split: {sorted(unassigned)}')
    return split


def volumes_for_split(data_root, split_name, split=None):
    """List the (image_path, label_path) pairs of every 3D volume in one split."""
    split = split or patient_split(data_root)
    patients = set(split[split_name])
    volumes = find_volumes(data_root)
    return [volumes[key] for key in sorted(volumes) if key[0] in patients]