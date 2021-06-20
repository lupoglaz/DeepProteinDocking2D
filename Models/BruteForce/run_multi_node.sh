#!/bin/bash
#SBATCH --partition=p_ccib_1
#SBATCH --job-name=ProteinDocking
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --constraint=volta
#SBATCH --time=48:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err
#SBATCH --export=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Bigger model
srun -N1 -n1 -w gpuc001 \
python distributed_train.py \
-experiment BigDatasetEBM \
-dataset_train BigDataset_60I_80A/SplitChains:training_set.dat \
-dataset_valid DockingDatasetSmall/SplitComplexes:validation_set.dat \
-batch_size_train 16 \
-max_decoys_train 8 \
-batch_size_valid 7 \
-max_epoch 20 \
-load 0 \
-gpus 2 \
-nodes $SLURM_NNODES \
-nr 0 &
srun -N1 -n1 -w gpuc002 \
python distributed_train.py \
q-experiment BigDatasetEBM \
-dataset_train BigDataset_60I_80A/SplitChains:training_set.dat \
-dataset_valid DockingDatasetSmall/SplitComplexes:validation_set.dat \
-batch_size_train 16 \
-max_decoys_train 8 \
-batch_size_valid 7 \
-max_epoch 20 \
-load 0 \
-gpus 2 \
-nodes $SLURM_NNODES \
-nr 1
