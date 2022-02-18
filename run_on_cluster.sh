#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=FI_BF_georgycode_20ep_batch8
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --constraint=volta
#SBATCH --time=48:00:00
#SBATCH --output=slurm_log/slurm.%N.%j.%x.out
#SBATCH --error=slurm_log/slurm.%N.%j.%x.err
#SBATCH --export=ALL

pwd
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

exp=$(echo FI_BF_georgycode_20ep_batch8)
python train_interaction.py -experiment "$exp" -docker -train -batch_size 8 -num_epochs 20
python train_docking.py -data_dir Log -experiment "$exp" -docker -test -gpu 0
