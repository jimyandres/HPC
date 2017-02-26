EXECS=mpi_matrix_mult
MPICC?=mpicc

all: ${EXECS}

mpi_matrix_mult: mpi_matrix_mult.c
	${MPICC} -o mpi_matrix_mult mpi_matrix_mult.c

clean:
	rm -f ${EXECS}