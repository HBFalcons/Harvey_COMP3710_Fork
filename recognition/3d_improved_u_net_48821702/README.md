# 3D Imrpoved U-Net Implementation for HipMRI Segmentation

## Algorithm and Problem

## Feasibility Review

### User Need, Scope, and Acceptance Criteria
The intended user would be prostate radiotherapists. The project works to segment HipMRI scans to classify the matter as prostate or non-prostate, such that targeted external-beam radiotherapy can be performed to treat the prostate only. There are negative impacts to misclassification in either direction, and the error margin is very small and so it is important to have highly accurate segmentations. Acceptance criteria will be:
1.	**Standard Global Dice Score**: Aim for ≥ 0.7
2.	**Slice-Wise Dice Profile**: A Dice score per slice, and can be compared across the model and baseline to show the specific regions of greatest improvements
3.	**Hausdorff Distance (HD95 specifically)**: This works to give make the statement that ‘95% of the estimated boundary is within {HD95} of the true boundary’. It determines the distance between the true and estimated boundaries
4.	**Craniocaudal Extent Error**: This is a signed difference between the estimated base and apex, and the true base and apexes of the boundary. These are the problem areas, and the sign determines what type of error occurs; under or overshooting.

### Model Choice and Course Concepts
The proposed dataset will be using HipMRI data. The proposed model will be a 3D improved U-net, with improvements from residual units and deep supervision. This fits into the **Hard** difficulty band and will use the baseline of a 2D U-net. A study by Yang et al. (2024) used multiple models for brain hippocampus segmentation on MR images and found the improved 3D U-net model with a filling technique to give better performance than the standard 3D U-net. The results showed an average Hausdorff distance of 5.1235 ± 1.4397, and a global Dice score of 0.7989 ± 0.0398. The improvements were different; however it brings motivation to trying other improvements and aiming for similar results.

### Preliminary Feasible Evidence
Within the HipMRI dataset, there are 211 3D MRI volumes from 38 patients, with labelled segmentations performed by experienced MR physicists. The segmentation is done into background, body, bone, bladder, rectum and prostate. 

### Risk, Budget and Fallback
The data houses an image input size of 256 x 256 x 128, with each voxel size being on average 1.68 x 1.68 x 1.56 mm. This is a limitation to the accuracy that the HD95 and extent error can give, and a risk would be that the improvements to the 3D U-net bring worse performance. The computation will be run on the Google Colab A100, and a fallback would be the standard 3D U-net ([Normal] difficulty) as this is known to work from Yang et al. (2024): https://pubmed.ncbi.nlm.nih.gov/38970025/ 


## Dependencies and Reproducibility

## Example Inputs, Outpus, and Plots

## Data Pre-Processing

## Open Research Dilemma

### Benchmarking

### Computational Efficiency Table

### Failure cases

### Engineering Recommendation

## Artificial Intelligence Usage Disclosure 