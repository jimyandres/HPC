#!/bin/bash
#
#SBATCH --job-name=mpi_matrix_mult
#SBATCH --output=res_mpi_matrix_mult.out
#SBATCH --ntasks=5
#SBATCH --nodes=5
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mpi_matrix_mult
