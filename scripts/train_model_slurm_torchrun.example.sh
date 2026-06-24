#!/bin/bash
# Example SLURM launch script for single-node, multi-GPU TERRA training (torchrun).
#
# Per-site SLURM scripts are git-ignored because they hold site-specific paths.
# Copy this template and adjust the SBATCH directives, environment activation,
# and GPU count for your cluster.
#
# NOTE: SBATCH directives are read by Slurm before the script runs and cannot
# expand shell variables, so set --output/--error to a path that exists on your
# site (here: a `logs/slurm/` directory relative to the submission directory).
#SBATCH --job-name=terra_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

export EXPERIMENT_NAME="my_experiment"
export RUN_NAME="run1"
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

nvidia-smi

# Activate your Python environment (edit for your site), e.g. with conda:
#   eval "$(conda shell.bash hook)" && conda activate terra
# ...or a virtualenv:
#   source "$TERRA_ENV_PATH/bin/activate"

# Load site paths (TERRA_REPO_DIR, TERRA_DATA_DIR, ...).
_TERRA_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_TERRA_SCRIPTS_DIR/cluster_env.sh" ] && source "$_TERRA_SCRIPTS_DIR/cluster_env.sh"

srun torchrun \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
  "$TERRA_REPO_DIR/src/terra/training/main.py" \
  --backend nccl \
  --fname "$TERRA_REPO_DIR/configs/model/my_experiment.yaml"
