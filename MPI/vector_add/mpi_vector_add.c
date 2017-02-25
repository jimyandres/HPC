#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void vec_add(int *A, int *B, int *ha_C, int COLS) {
	for(int i=0;i<COLS;i++){
		ha_C[i] = A[i] + B[i];
	}
}

int main(int argc, char **argv) {
	double t1, t2;
	int *A, *B, *ha_C, *da_C, *p_A, *p_B, *p_C;
	int size, count, COLS;

	if(argc == 2) 
		COLS = (int*)argv[1];
	else
		return 1;

	size = COLS * sizeof(int);

	A = (int*)malloc(size);
 	B = (int*)malloc(size);
 	ha_C = (int*)malloc(size);	//Host answer
 	da_C = (int*)malloc(size);	//Device answer

	for(int i=0;i<COLS;i++) {
		A[i]=1;
		B[i]=2;
	}

	//Initialize the MPI environment
	MPI_Init(NULL,NULL);

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


	/*******************CPU*******************/

	t1 = MPI_Wtime();

	vec_add(A,B,ha_C, COLS);

	t2 = MPI_Wtime();

	/*******************END*******************/

	printf("CPU: %f\n", t2-t1);

	for (int i = 0; i < COLS; ++i) {
		printf("%d, ", ha_C[i]);
	}

	/*******************MPI*******************/

 	count = COLS/world_size;
 	p_A = (int*)malloc(count);
 	p_B = (int*)malloc(count);
 	p_C = (int*)malloc(count);


	t1 = MPI_Wtime();

	MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Send A
	MPI_Scatter(A, count, MPI_INT, p_A, count, MPI_INT, 0, MPI_COMM_WORLD);

	//Send B
	MPI_Scatter(B, count, MPI_INT, p_B, count, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < count; ++i)	{
		p_C[i] = p_A[i] + p_A[i];
	}

	//Take results
	MPI_Gather(p_C, count, MPI_INT, da_C, count, MPI_INT, 0, MPI_COMM_WORLD);

	t2 = MPI_Wtime();

	/*******************END*******************/

	printf("GPU: %f\n", t2-t1);

	for (int i = 0; i < COLS; ++i) {
		printf("%d, ", da_C[i]);
	}

	MPI_Finalize();
}