#!/bin/bash
#SBATCH --job-name=clean_data_0.75
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --output=clean_data_%j.out
#SBATCH --error=clean_data_%j.err

set -e

rm -rf /scratch/justincb/bluesky_faces_cleaned_75_percent
mkdir -p /scratch/justincb/bluesky_faces_cleaned_75_percent

module load python/3.10
module load cuda/12.2

source ~/envs/tf/bin/activate

cd /home/justincb/code

python interesting_data_0.75.py