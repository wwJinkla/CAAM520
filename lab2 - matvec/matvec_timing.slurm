#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --job-name=matvec_timing
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:01:00
#SBATCH --mail-user=weiwu995@gmail.com
#SBATCH --mail-type=ALL

srun -n 1 ./matvec2 2048
srun -n 2 ./matvec2 1024
srun -n 4 ./matvec2 512
srun -n 8 ./matvec2 256
