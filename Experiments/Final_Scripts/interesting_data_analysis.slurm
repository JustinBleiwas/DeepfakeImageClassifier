#!/bin/bash
#SBATCH --job-name=interesting_data_analysis
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=interesting_data_analysis_%j.out
#SBATCH --error=interesting_data_analysis_%j.err

echo "Copying dataset to local node storage..."
cp -r /scratch/justincb/bluesky_faces_cleaned_50_percent $SLURM_TMPDIR/
cp -r /scratch/justincb/bluesky_faces_cleaned_75_percent $SLURM_TMPDIR/
cp -r /scratch/justincb/bluesky_faces_cleaned_100_percent $SLURM_TMPDIR/
echo "Copy complete."

echo "Fixing directory structure..."

mkdir $SLURM_TMPDIR/bluesky_faces_cleaned_50_percent/dummy
mv $SLURM_TMPDIR/bluesky_faces_cleaned_50_percent/*.jpg \
   $SLURM_TMPDIR/bluesky_faces_cleaned_50_percent/dummy/

mkdir $SLURM_TMPDIR/bluesky_faces_cleaned_75_percent/dummy
mv $SLURM_TMPDIR/bluesky_faces_cleaned_75_percent/*.jpg \
   $SLURM_TMPDIR/bluesky_faces_cleaned_75_percent/dummy/

mkdir $SLURM_TMPDIR/bluesky_faces_cleaned_100_percent/dummy
mv $SLURM_TMPDIR/bluesky_faces_cleaned_100_percent/*.jpg \
   $SLURM_TMPDIR/bluesky_faces_cleaned_100_percent/dummy/

echo "Structure fixed."

module load python/3.10
module load cuda/12.2

source ~/envs/tf/bin/activate

cd /home/justincb/code

python interesting_data_analysis.py