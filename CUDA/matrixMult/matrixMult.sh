#!/bin/bash
#
#SBATCH --job-name=matrixMult
#SBATCH --output=res_matrixMult.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

for (( i = 500; i <= 2000; i+500 )); do
	for (( j = 0; j < 20; j++ )); do
		srun matrixMult $i
	done
done

