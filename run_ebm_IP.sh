#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=EBM_IP_1LDsamp_100ep_noGS
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

exp=$(echo EBM_IP_1LDsamp_100ep_noGS)

#EBM: 1D sample buffer, no add_positive
srun -N1 -n1 python -u train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples -no_global_step >> slurm_log/"$exp"_train_prints.out
srun -N1 -n1 python -u results.py -experiment "$exp" -docking -max_epoch 100 >> slurm_log/"$exp"_results_prints.out
srun -N1 -n1 python -u train_docking.py -experiment "$exp" -test -ebm -gpu 0 >> slurm_log/"$exp"_test_prints.out

#exp=$(echo EBM_IP_1LDsamp_100ep_withGS)
#
##EBM: 1D sample buffer, no add_positive
#srun -N1 -n1 python -u train_docking.py -experiment "$exp" -train -ebm -num_epochs 100 -batch_size 1 -gpu 0 -num_samples 1 -no_pos_samples >> slurm_log/"$exp"_train_prints.out
#srun -N1 -n1 python -u results.py -experiment "$exp" -docking -max_epoch 100 >> slurm_log/"$exp"_results_prints.out
#srun -N1 -n1 python -u train_docking.py -experiment "$exp" -test -ebm -gpu 0 >> slurm_log/"$exp"_test_prints.out