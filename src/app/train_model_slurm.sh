#!/bin/bash
#SBATCH --job-name=nichejepa_train
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --constraint=h100_80gb  # Request specific GPU type
#SBATCH --mem-per-gpu=64G      # Memory per GPU
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00         # Walltime
#SBATCH --output=logs/%x_%j.out # Log file with job name and ID
#SBATCH --error=logs/%x_%j.err  # Error log

# Load your Python environment (e.g., conda or module load)
source ~/.bashrc
conda activate nichejepa

# Optional: Set master port if using TCP init method
MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT

# Set experiment identifiers (optional: used in your script)
export EXPERIMENT_NAME="hst_corpus_70m"
export RUN_NAME="gtsmall_combined_1"

# Run the distributed training using torchrun (preferred)
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /home/aih/sebastian.birk/workspace/projects/nichejepa/src/app/main_dist.py --backend nccl --fname /home/aih/sebastian.birk/workspace/projects/nichejepa/configs/model/hst_corpus_70m_gt_small.yaml