#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=hw03
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:30:00
#SBATCH --mail-user=weiwu995@gmail.com
#SBATCH --mail-type=ALL

srun ./hw03 32 1e-6 1
srun ./hw03 32 1e-6 2
srun ./hw03 32 1e-6 4
srun ./hw03 32 1e-6 8
srun ./hw03 32 1e-6 16


srun ./hw03 64 1e-6 1
srun ./hw03 64 1e-6 2
srun ./hw03 64 1e-6 4
srun ./hw03 64 1e-6 8
srun ./hw03 64 1e-6 16


srun ./hw03 128 1e-6 1
srun ./hw03 128 1e-6 2
srun ./hw03 128 1e-6 4
srun ./hw03 128 1e-6 8
srun ./hw03 128 1e-6 16
