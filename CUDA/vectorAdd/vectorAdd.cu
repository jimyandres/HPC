#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#define TAM 5

void llenarVector(int *A) {
	//srand(time(NULL));
	for(int i=0; i<TAM; i++) {
		A[i]=rand();
	}
}

__global__ void sumaVectores(int *A, int *B, int *C) {
	int i = threadIdx.x+blockDim.x * blockIdx.x;
	if(i<TAM)
		C[i] = A[i]+B[i];
}

void printVector(int *A) {
	printf("(");
	for(int i=0; i<TAM; i++) {
		printf("%d ", A[i]);
		if(i!=TAM-1) {
			printf(", ");
		}
	}
	printf(")\n");
}

int main(){

	int size = TAM*sizeof(int);

	int *A = (int *) malloc(size);
	int *B = (int *) malloc(size);
	int *C = (int *) malloc(size);

	int *d_A, *d_B, *d_C;
	cudaError_t err = cudaMalloc((void**)&d_A,size);
	if (err != cudaSuccess) {
		printf("Error %s", cudaGetErrorString( err));
		exit(EXIT_FAILURE);
	}	

	err = cudaMalloc((void**)&d_B,size);
	if (err != cudaSuccess) {
                printf("Error %s", cudaGetErrorString( err));
                exit(EXIT_FAILURE);
        }

	err = cudaMalloc((void**)&d_C,size);
	if (err != cudaSuccess) {
                printf("Error %s", cudaGetErrorString( err));
                exit(EXIT_FAILURE);
        }

	llenarVector(A);
//	printVector(A);
	llenarVector(B);
//	printVector(B);

	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

	sumaVectores<<<ceil(TAM/64),64>>>(d_A,d_B,d_C);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost );

	printVector(C);
	//printf( "c[0] = %d\n",0,C[0] );
	//printf( "c[%d] = %d\n",TAM-1, C[TAM-1] );

	err = cudaFree(d_A);
	if (err != cudaSuccess) {
                printf("Error %s", cudaGetErrorString( err));
                exit(EXIT_FAILURE);
        }

	err = cudaFree(d_B);
	if (err != cudaSuccess) {
                printf("Error %s", cudaGetErrorString( err));
                exit(EXIT_FAILURE);
        }

	err = cudaFree(d_C);
	if (err != cudaSuccess) {
                printf("Error %s", cudaGetErrorString( err));
                exit(EXIT_FAILURE);
        }

	free(A);
	free(B);
	free(C);
	return 0;

}
