#!/bin/bash
#SBATCH --job-name=srm_hparam
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=srm_hparam_%j.out
#SBATCH --error=srm_hparam_%j.err

echo "Copying dataset to local node storage..."
cp -r /scratch/justincb/merged_dataset $SLURM_TMPDIR/
cp -r /scratch/justincb/test_dataset $SLURM_TMPDIR/
echo "Copy complete."

module load python/3.10
module load cuda/12.2

source ~/envs/tf/bin/activate

cd /home/justincb/code

python Merged_Dataset_SRM_Tuning_Script_Refined_AUC.py