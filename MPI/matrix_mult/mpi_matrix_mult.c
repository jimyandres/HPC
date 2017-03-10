#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

#define NRA 1000
#define NCA 1000 
#define NCB 1000

int main(int argc, char *argv[]) {
	double t1, t2;
	int *p_A, *p_C;
	int count;

	int A[NRA+4][NCA], B[NCA][NCB], C[NRA][NCB];

	//Initialize the MPI environment
	MPI_Init(&argc, &argv);

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(world_rank == MASTER){	

		/*******************MPI*******************/
	
	 	count = NRA/world_size;

		if (NRA%world_size != 0) {
			count += 2;
			for(int i=0;i<(count*world_size-NRA);i++)
				A[NRA][i] = 0;
		}

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

	 	p_A = (int*)malloc(sizeof(int*)*NCA*count);
//	 	p_B = (int*)malloc(sizeof(int*)*NCA*NCB);
	 	p_C = (int*)malloc(sizeof(int*)*count*NCB);

		t1 = MPI_Wtime();

		MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//Send A
		MPI_Scatter(A, NCA*count, MPI_INT, p_A, NCA*count, MPI_INT, 0, MPI_COMM_WORLD);

		//Send B
		MPI_Bcast(&B, NCA*NCB, MPI_INT, 0, MPI_COMM_WORLD);
		for (int c = 0; c < count; ++c) {
			for (int k = 0; k < NCB; ++k) {
				p_C[c*NCB+k] = 0;
				for (int j = 0; j < NCA; ++j) {
					p_C[c*NCB+k] += p_A[c*NCA+j] * B[j][k];
				}
			}
		}
		//Take results
		MPI_Gather(p_C, count*NCB, MPI_INT, C, count*NCB, MPI_INT, 0, MPI_COMM_WORLD);

		t2 = MPI_Wtime();

		/*******************END*******************/

		/*for (int i = 0; i < NCB; ++i) {
			for (int j = 0; j < NRA; ++j)
			{
				printf("%d, ", C[i][j]);
			}
			printf("\n");
		}*/
		printf("\nGPU: %f\n", t2-t1);
		printf("Done.\n");
	}

	if (world_rank > MASTER) {

		MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		p_A = (int*)malloc(sizeof(int*)*NCA*count);
//                p_B = (int*)malloc(sizeof(int*)*NCA*NCB);
	        p_C = (int*)malloc(sizeof(int*)*count*NCB);
		
		//Send A
		MPI_Scatter(A, NCA*count, MPI_INT, p_A, NCA*count, MPI_INT, 0, MPI_COMM_WORLD);

		//Send B
		MPI_Bcast(&B, NCA*NCB, MPI_INT, 0, MPI_COMM_WORLD);

		for (int c = 0; c < count; ++c) {
			for (int k = 0; k < NCB; ++k) {
				p_C[c*NCB+k] = 0;
				for (int j = 0; j < NCA; ++j) {
					p_C[c*NCB+k] += p_A[c*NCA+j] * B[j][k];
				}
			}
		}
		
		//Take results
		MPI_Gather(p_C, count*NCB, MPI_INT, C, count*NCB, MPI_INT, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
