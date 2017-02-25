#include <mpi.h>
#include <stdio.h>

#define ROWS 1000

void vec_add(int *A, int *B, int *h_C) {
	for(int i=0;i<ROWS;i++){
			h_C[i] = A[i] + B[i];
	}
}

int main(int argc, char **argv) {
	double t1, t2;
	int *A, *B, *h_C, *d_C;
	int size = ROWS * sizeof(int);

	A = (int*)malloc(size);
 	B = (int*)malloc(size);
 	h_C = (int*)malloc(size);
 	d_C = (int*)malloc(size);

	for(int i=0;i<ROWS;i++){
			A[i]=1;
			B[i]=2;
	}

	//------CPU

	t1 = MPI_Wtime();

	vec_add(A,B,h_C);

	t2 = MPI_Wtime();

	printf("CPU: %f\n", t2-t1);

}