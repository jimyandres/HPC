#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define COLS 100

int main(int argc, char *argv[]) {
	double t1, t2;
	int *A, *B, *ha_C, *da_C, *p_A, *p_B, *p_C;
	int size, count;

	//Initialize the MPI environment
	MPI_Init(&argc, &argv);

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(world_rank == MASTER){	

		A = (int*)malloc(size);
	    B = (int*)malloc(size);
		ha_C = (int*)malloc(size);      //Host answer
	    da_C = (int*)malloc(size);      //Device answer

        for(int i=0;i<COLS;i++) {
                A[i]=1;
    	        B[i]=2;
        }

		/*******************MPI*******************/
	
	 	count = COLS/world_size;

		if (COLS%world_size != 0) {
			count += 1;
			for(int i=0;i<(count*world_size-COLS);i++)
				A[COLS+i] = B[COLS+i] = 0;
		}

	 	p_A = (int*)malloc(sizeof(int*)*count);
	 	p_B = (int*)malloc(sizeof(int*)*count);
	 	p_C = (int*)malloc(sizeof(int*)*count);

		t1 = MPI_Wtime();

		MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//Send A
		MPI_Scatter(A, count, MPI_INT, p_A, count, MPI_INT, 0, MPI_COMM_WORLD);

		//Send B
		MPI_Scatter(B, count, MPI_INT, p_B, count, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < count; ++i)	{
			p_C[i] = p_A[i] + p_B[i];
		}

		//Take results
		MPI_Gather(p_C, count, MPI_INT, da_C, count, MPI_INT, 0, MPI_COMM_WORLD);

		t2 = MPI_Wtime();

		/*******************END*******************/

		for (int i = 0; i < COLS; ++i) {
			printf("%d, ", da_C[i]);
		}
		printf("\nGPU: %f\n", t2-t1);
		printf("Done.\n");
	}

	else {

		MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		p_A = (int*)malloc(sizeof(int*)*count);
	 	p_B = (int*)malloc(sizeof(int*)*count);
	 	p_C = (int*)malloc(sizeof(int*)*count);

		//Send A
		MPI_Scatter(A, count, MPI_INT, p_A, count, MPI_INT, 0, MPI_COMM_WORLD);

		//Send B
		MPI_Scatter(B, count, MPI_INT, p_B, count, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < count; i++)	{
			p_C[i] = p_A[i] + p_B[i];
		}

		//Take results
		MPI_Gather(p_C, count, MPI_INT, da_C, count, MPI_INT, 0, MPI_COMM_WORLD);

	}

	MPI_Finalize();
	return 0;
}
