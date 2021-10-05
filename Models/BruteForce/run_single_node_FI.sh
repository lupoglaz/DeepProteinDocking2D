#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=scratch_rep1_ones
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --constraint=volta
#SBATCH --time=48:00:00
#SBATCH --output=slurm_log/slurm.%N.scratch_rep1_ones.out
#SBATCH --error=slurm_log/slurm.%N.scratch_rep1_ones.err
#SBATCH --export=ALL

pwd
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -N1 -n1 python train_bruteforce_interaction.py rep1;

#srun -N1 -n1 python train_bruteforce_interaction.py rep2;

#srun -N1 -n1 python train_bruteforce_interaction.py rep3;


#srun -N1 -n1 \
#python distributed_train.py \
#-experiment SmallDataset \
#-dataset_train DockingDatasetSmall/SplitChains:training_set.dat \
#-dataset_valid DockingDatasetSmall/SplitComplexes:validation_set.dat \
#-batch_size_train 18 \
#-max_decoys_train 9 \
#-batch_size_valid 7 \
#-max_epoch 300 \
#-load 0 \
#-gpus 2 \
#-nodes $SLURM_NNODES \
#-nr 0
