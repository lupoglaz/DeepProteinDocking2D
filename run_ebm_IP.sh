#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=EBM_IP_1LDsamp_100ep_withGS_hot
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

#exp=$(echo EBM_IP_1LDsamp_100ep_noGS)
#
##EBM: 1D sample buffer, no add_positive
#python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples -no_global_step
#python results.py -experiment "$exp" -docking -max_epoch 100
#python train_docking.py -experiment "$exp" -test -ebm -gpu 0

#exp=$(echo EBM_IP_1LDsamp_100ep_withGS)
#
##EBM: 1D sample buffer, no add_positive
#python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples
#python results.py -experiment "$exp" -docking -max_epoch 100
#python train_docking.py -experiment "$exp" -test -ebm -gpu 0

#exp=$(echo EBM_IP_1LDsamp_100ep_parallel_warmhotsigma_noGSnoAP)
#
##EBM: 1D sample buffer, no add_positive
#python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -parallel_noGSAP
#python results.py -experiment "$exp" -docking -max_epoch 100
#python train_docking.py -experiment "$exp" -test -ebm -gpu 0

#exp=$(echo EBM_IP_1LDsamp_100ep_parallel_warmhotsigma_withGSnoAP)
#
##EBM: 1D sample buffer, no add_positive
#python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -parallel
#python results.py -experiment "$exp" -docking -max_epoch 100
#python train_docking.py -experiment "$exp" -test -ebm -gpu 0

#exp=$(echo EBM_IP_1LDsamp_100ep_noGS_hot)
#
##EBM: 1D sample buffer, no add_positive
#python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples -no_global_step
#python results.py -experiment "$exp" -docking -max_epoch 100
#python train_docking.py -experiment "$exp" -test -ebm -gpu 0

exp=$(echo EBM_IP_1LDsamp_100ep_withGS_hot)

#EBM: 1D sample buffer, no add_positive
srun -N1 -n1 python train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples
srun -N1 -n1 python results.py -experiment "$exp" -docking -max_epoch 100
srun -N1 -n1 python train_docking.py -experiment "$exp" -test -ebm -gpu 0