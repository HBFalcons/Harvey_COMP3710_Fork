# Data loading, splitting, and augmentation for the HipMRI dataset
# Based on dataset, splitting is done at the patient level, so that all slices from a given patient are either in the training or validation set.

import math
import os
import re
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader



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


# ---------------------------------------------------------------------------
# NIfTI loading - as per the COMP3710 task sheet
# Only change being removing unused 'orient' and 'interp' variables
# In current, pipeline use, no issues with functions performance
# ---------------------------------------------------------------------------
EXPECTED_SHAPE_2D = (256, 128)
EXPECTED_SHAPE_3D = (256, 256, 128)


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def to_one_hot(labels, num_classes=NUM_CLASSES):
    """Convert class-index labels (B, *spatial) into one-hot channels (B, C, *spatial).

    Done on the GPU inside training/evaluation rather than at load time: it always
    produces num_classes channels (unlike to_channels above), and pre-loading one-hot
    3D labels would need ~6x more memory than integer label maps.
    """
    return F.one_hot(labels, num_classes).movedim(-1, 1).float()


# ---------------------------------------------------------------------------
# PyTorch datasets (each split is pre-loaded into memory with the loaders above)
# ---------------------------------------------------------------------------
class HipMRISlices2D(Dataset):
    """2D coronal slices from keras_slices_data, one slice per item (for the 2D U-Net)."""

    def __init__(self, data_root, split_name):
        cases = find_slices(data_root, split_name)
        # Flatten to (case_key, slice_index, image_path, mask_path) in a fixed order,
        # so item i can always be traced back to its patient, week and slice
        self.items = [(key, idx, img, msk)
                      for key in sorted(cases)
                      for idx, img, msk in cases[key]]
        # normImage=True applies z-score normalisation (despite the "0.0-1.0" docstring)
        self.images = load_data_2D([img for _, _, img, _ in self.items], normImage=True)
        self.labels = load_data_2D([msk for _, _, _, msk in self.items], dtype=np.uint8)
        if self.images.shape[1:] != EXPECTED_SHAPE_2D:
            raise ValueError(f'Unexpected 2D slice shape {self.images.shape[1:]}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[i]).unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(self.labels[i]).long()        # (H, W), values 0..5
        return image, label


class HipMRIVolumes3D(Dataset):
    """Full 3D MR volumes with their label volumes, one volume per item (for the 3D model)."""

    def __init__(self, data_root, split_name, split=None):
        self.pairs = volumes_for_split(data_root, split_name, split)
        # normImage=True applies z-score normalisation (despite the "0.0-1.0" docstring)
        self.images = load_data_3D([img for img, _ in self.pairs], normImage=True)
        self.labels = load_data_3D([lab for _, lab in self.pairs], dtype=np.uint8)
        if self.images.shape[1:] != EXPECTED_SHAPE_3D:
            raise ValueError(f'Unexpected 3D volume shape {self.images.shape[1:]}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[i]).unsqueeze(0)  # (1, D, H, W)
        label = torch.from_numpy(self.labels[i]).long()        # (D, H, W), values 0..5
        return image, label


def get_dataloader(data_root, split_name, mode, batch_size, num_workers=4):
    """Build a DataLoader for one split.

    mode: '2d' for single slices, '3d' for whole volumes.
    Only the training split is shuffled; validation/test keep a fixed order.
    """
    if mode == '2d':
        dataset = HipMRISlices2D(data_root, split_name)
    elif mode == '3d':
        dataset = HipMRIVolumes3D(data_root, split_name)
    else:
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")

    is_train = split_name == 'train'
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=is_train,
                      drop_last=is_train,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())