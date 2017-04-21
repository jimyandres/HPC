#!/bin/bash
#
#SBATCH --job-name=2_ImageConvolution
#SBATCH --output=res_2_ImageConvolution.md
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1


declare -a array=("test2.png" "test1.png" "test3.jpg")

for i in "${array[@]}"
do
	echo "**Image** = $i"
	echo
	echo "| n | Sequential on Host | Sobel on Host | Acceleration | Sequential on Device | Acceleration | Sobel on Device | Acceleration |"
	echo "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
	for (( j = 1; j <= 20; j++ )); do
		echo -n "| $j | "
		srun 2_ImageConvolution $i seq_h sobel_h seq_d sobel_d 
	done
	echo
done
