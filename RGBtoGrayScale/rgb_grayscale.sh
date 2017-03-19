#!/bin/bash
#
#SBATCH --job-name=rgb_grayscale
#SBATCH --output=rgb_grayscale.out
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun rgb_grayscale img.png 1
