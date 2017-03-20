#!/bin/bash
#
#SBATCH --job-name=rgb_grayscale
#SBATCH --output=res_rgb_grayscale.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

srun rgb_grayscale img.png 20 
