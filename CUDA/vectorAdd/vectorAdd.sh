#!/bin/bash
#
#SBATCH --job-name=vectorAdd
#SBATCH --output=res_vectorAdd.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES
mpirun vectorAdd
