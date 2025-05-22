#!/bin/bash
#SBATCH --job-name=my_distributed_training
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1                    # Number of tasks (1 task for distributed)
#SBATCH --cpus-per-task=8             # CPUs per task (adjust based on requirements)
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --mem=256G                    # Total memory (adjust as needed)
#SBATCH --time=24:00:00               # Wall time
#SBATCH --partition=gpu_p             # GPU partition (adjust based on available queues)
#SBATCH --qos=gpu_normal              # Quality of service for GPUs
#SBATCH --exclusive                   # Ensure exclusive access to nodes

# Load necessary modules
module load cuda/11.3                # Load CUDA (adjust based on your setup)
module load python/3.8                # Load Python (adjust as needed)
module load anaconda/3                # Load Anaconda (if using a conda environment)

# Activate your Python virtual environment (if needed)
source activate your_environment_name  # Replace with your virtual environment name

# Set distributed training environment variables
export MASTER_ADDR=$(hostname)         # Set master address as the node name
export MASTER_PORT=12345               # Set an available port for communication
export WORLD_SIZE=4                    # Total number of processes (equal to the number of GPUs)
export RANK=0                          # This will be set dynamically for each process
export LOCAL_RANK=0                    # This will be set dynamically for each GPU
export SLURM_NTASKS=4                  # Set the number of tasks for distributed training

# Run the Python script
srun python main_dist.py --backend nccl --fname configs.yaml
