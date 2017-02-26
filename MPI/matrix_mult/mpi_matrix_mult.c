#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

#define NRA 5
#define NCA 5
#define NCB 5

int main(int argc, char *argv[]) {
	double t1, t2;
	int *p_A, *p_B, *p_C;
	int count;


	int A[NRA][NCA], B[NCA][NCB], C[NRA][NCB];

	//Initialize the MPI environment
	MPI_Init(&argc, &argv);

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(world_rank == MASTER){	

        for(int i=0;i<NRA;i++) {
        	for (int j = 0; j < NCA; ++j) {
                A[i][j] = 1;
        	}
        }

        for(int i=0;i<NCA;i++) {
        	for (int j = 0; j < NCB; ++j) {
                B[i][j] = 2;
        	}
        }

		/*******************MPI*******************/
	
	 	count = NCA/world_size;

		if (NCA%world_size != 0) {
			count += 1;
			for(int i=0;i<(count*world_size-NCA);i++)
				A[i][NCA+i] = B[NCA+i][i] = 0;
		}

	 	p_A = (int*)malloc(sizeof(int*)*count);
	 	p_B = (int*)malloc(sizeof(int*)*count*NCB);
	 	p_C = (int*)malloc(sizeof(int*)*count);

		t1 = MPI_Wtime();

		for (int i = 0; i < count; ++i) {

			MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

			//Send A
			MPI_Scatter(A[i], count, MPI_INT, p_A, count, MPI_INT, 0, MPI_COMM_WORLD);

			//Send B
			MPI_Scatter(B, count, MPI_INT, p_B, count, MPI_INT, 0, MPI_COMM_WORLD);

			for (int c = 0; c < NCB; ++c) {
				p_C[c] = 0;
				for (int j = 0; j < count; ++j)
				{
					p_C[c] += p_A[j] * p_B[j][c];
				}				
			}

			//Take results
			MPI_Gather(p_C, count, MPI_INT, C[i], count, MPI_INT, 0, MPI_COMM_WORLD);
		}

		t2 = MPI_Wtime();

		/*******************END*******************/

		for (int i = 0; i < NCB; ++i) {
			for (int j = 0; j < NCB; ++j)
			{
				printf("%d, ", C[i][j]);
			}
			printf("\n");
		}
		printf("\nGPU: %f\n", t2-t1);
		printf("Done.\n");
	}

	else {

		for (int i = 0; i < count; ++i) {
			MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

			p_A = (int*)malloc(sizeof(int*)*count);
		 	p_B = (int*)malloc(sizeof(int*)*count);
		 	p_C = (int*)malloc(sizeof(int*)*count);

			//Send A
			MPI_Scatter(A, count, MPI_INT, p_A, count, MPI_INT, 0, MPI_COMM_WORLD);

			//Send B
			MPI_Scatter(B, count, MPI_INT, p_B, count, MPI_INT, 0, MPI_COMM_WORLD);

			for (int c = 0; c < NCB; ++c) {
				p_C[c] = 0;
				for (int j = 0; j < count; ++j)
				{
					p_C[c] += p_A[j] * p_B[j][c];
				}				
			}

			//Take results
			MPI_Gather(p_C, count, MPI_INT, C, count, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}
