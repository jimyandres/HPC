#!/bin/bash
#
#SBATCH --job-name=matrixMult_V2
#SBATCH --output=res_matrixMult_V2.csv
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

for i in {10,50}; do
	echo "N = $i",,,
	for (( j = 0; j < 20; j++ )); do
		echo -n "$j,"
		srun matrixMult_v2 $i 
	done
done

