#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --job-name=hw02
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:30:00
#SBATCH --mail-user=weiwu995@gmail.com
#SBATCH --mail-type=ALL


srun -n 1./hw02 16 1e-6 
srun -n 2./hw02 16 1e-6 
srun -n 4./hw02 16 1e-6 
srun -n 8./hw02 16 1e-6 

srun -n 1./hw02 32 1e-6 
srun -n 2./hw02 32 1e-6 
srun -n 4./hw02 32 1e-6 
srun -n 8./hw02 32 1e-6 

srun -n 1./hw02 64 1e-6 
srun -n 2./hw02 64 1e-6 
srun -n 4./hw02 64 1e-6 
srun -n 8./hw02 64 1e-6 

