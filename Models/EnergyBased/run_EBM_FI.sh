#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=EBM_MC_FI
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

srun -N1 -n1 python train_energybased_interaction.py;
