#!/usr/bin/env bash

# Navigate to proper directory and activate the conda environment
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..
source activate gym

# Log directory is based on run date
DATE=$(date +%Y-%m-%d)

# Run several different training runs, storing in different log locations
NAME="experiment_name"
python train.py --log=$DATE$NAME \
--num_train_cycles=1 \
--buffer_size=1 \
--num_trajectories=1 \
--num_learning_iterations=1 \
--episode_batch_size=1 \
--batch_norm \
--loss=discounted_rewards \
--non_linear=relu