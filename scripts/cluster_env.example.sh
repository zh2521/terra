#!/usr/bin/env bash
# TERRA site configuration (TEMPLATE).
#
# Copy this file to `cluster_env.sh` (which is git-ignored) and set the paths for
# YOUR environment:
#
#     cp scripts/cluster_env.example.sh scripts/cluster_env.sh
#     # then edit scripts/cluster_env.sh
#
# The launch scripts in this folder source `cluster_env.sh`, and the training
# config loader expands these variables inside YAML configs. For example, a
# model-config value of "${TERRA_DATA_DIR}/silver/my_dataset" resolves against
# TERRA_DATA_DIR at load time. To use the variables in an interactive shell too:
#
#     source scripts/cluster_env.sh

# Root directory holding the (primary) tokenized / raw datasets.
export TERRA_DATA_DIR="/path/to/DATASETS"

# Optional secondary dataset root (e.g. a different filesystem or species).
export TERRA_DATA_DIR_ALT="/path/to/other/DATASETS"

# Directory for run artifacts: checkpoints, logs, norm-factor / pf-target files.
export TERRA_ARTIFACTS_DIR="/path/to/artifacts"

# Local clone of the TERRA repository.
export TERRA_REPO_DIR="/path/to/terra"

# Python environment to activate for training / inference jobs.
export TERRA_ENV_PATH="/path/to/venv"
