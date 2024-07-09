#!/bin/bash
#SBATCH --partition="HPC-4GPUs,HPC-8GPUs"
#SBATCH --mem="16G"
#SBATCH --cpus-per-task="4"
#SBATCH --gres=gpu:1
#SBATCH --time="24:00:00"
#SBATCH --output=/home/rassmanns/diffusion/flairsyn/output/logs/slurm/%j_%x.out
#SBATCH --job-name="wmh_pred_n4_2"

module load singularity
cd /home/$USER/diffusion/SHIVA_WMH

# predict all original validation samples after bias-field correction
sh run_predict_from_out_dir_singularity.sh ../flairsyn/output/original/inference pred_flair_n4.nii.gz

