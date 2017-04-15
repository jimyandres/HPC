#!/bin/bash
#
#SBATCH --job-name=ImageConvolution
#SBATCH --output=res_ImageConvolution.md
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1


i="test1.png"

echo "**Image** = $i"
echo
echo "| n | Sequential on Host | Sobel on Host | Acceleration | Sequential on Device | Acceleration | Sobel on Device | Acceleration |"
echo "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
for (( j = 1; j <= 20; j++ )); do
	echo -n "| $j | "
	srun ImageConvolution $i seq_h sobel_h seq_d sobel_d 
done
echo
