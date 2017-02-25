#!/bin/bash
#
#SBATCH --job-name=mpi_vector_add
#SBATCH --output=res_mpi_vector_add.out
#SBATCH --ntasks=4
#SBATCH --nodes=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mpi_vector_add
