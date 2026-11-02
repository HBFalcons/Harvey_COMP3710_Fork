#!/bin/bash
# Train and evaluate one model on Rangpur.
# Usage (from this folder):
#   sbatch run_rangpur.sh unet2d                       # full 2D baseline
#   sbatch run_rangpur.sh improved_unet3d              # full 3D model
#   sbatch run_rangpur.sh unet2d ./outputs/test --epochs 1   # quick test run
#SBATCH --job-name=hipmri_seg
#SBATCH --partition=comp3710           # check available GPU partitions with `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out         # log file in the submission folder

MODEL=${1:-unet2d}
OUT_DIR=${2:-./outputs/$MODEL}
DATA_DIR=/home/groups/comp3710/HipMRI_Study_open

# Activate the environment with requirements.txt installed
eval "$(conda shell.bash hook)"
conda activate comp3710

python train.py --model "$MODEL" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" "${@:3}"
python evaluation.py --checkpoint "$OUT_DIR/best.pt" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR/eval"