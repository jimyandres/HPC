EXECS=mpi_vector_add
MPICC?=mpicc

all: ${EXECS}

mpi_vector_add: mpi_vector_add.c
	${MPICC} -o mpi_vector_add mpi_vector_add.c

clean:
	rm -f ${EXECS}
