#!/bin/bash
#
#SBATCH --job-name=matrixMult_V2
#SBATCH --output=res_matrixMult_V2.md
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

for i in {100,500,1000,1500}; do
	echo "**N** = $i"
	echo
	echo "| n | Serial | CUDA w/o SharedMem | Acceleration | CheckResult | CUDA w/ SharedMem | Acceleration | CheckResult |"
	echo "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
	for (( j = 0; j < 20; j++ )); do
		echo -n "| $j | "
		srun matrixMult_V2 $i s c ct 
	done
	echo
done
