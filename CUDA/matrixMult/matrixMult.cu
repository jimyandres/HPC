#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


__global__ void matrixMultGPU (int *A,int *B,int *C, int N){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int acc;
	if(col < N && row < N){
    acc = 0;
		for(int k=0;k<N;k++)
			acc += A[row*N+k] * B[k*N+col];
		C[row*N+col] = acc;
	}
}

void matrixMultCPU(int *A,int *B,int *C, int N){
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int acc=0;
			for(int k=0;k<N;k++){
				int m = A[i*N+k];
				int n = B[k*N+j];
				acc += m*n;
			}
		C[i*N+j] = acc;
		}
	}
}

int main(int argc, char **argv){
	
	long *A, *B, *C1, *C2;
	long *d_A, *d_B, *d_C;

	if(argc != 2) {
		printf("No size given\n");
		return -1;
	}
	long N = strtol(argv[2], NULL, 10);

	long size = N*N*sizeof(long);

  	A = (long*)malloc(size);
 	B = (long*)malloc(size);
 	C1 = (long*)malloc(size);
 	C2 = (long*)malloc(size);

	for(int i=0;i<N*N;i++){
			A[i]=1.0;
			B[i]=2.0;
	}

	//CPU----------------------------
	clock_t tic = clock();
	matrixMultCPU(A,B,C1, N);
  	clock_t toc = clock();
	printf("Tiempo CPU: %f segundos\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	//-------------------------------
  
	cudaMalloc((void**)&d_A,size);
	cudaMalloc((void**)&d_B,size);
	cudaMalloc((void**)&d_C,size);

	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

	//GPU----------------------------
	dim3 dimBlock(32,32);
  	dim3 dimGrid(ceil(N/float(dimBlock.x)),ceil(N/float(dimBlock.y)),1);
	
  	clock_t tic = clock();
	matrixMultGPU<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);
  	//cudaDeviceSynchronize();
	cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
  	clock_t toc = clock();
	printf("\n\nTiempo: %f segundos\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	//--------------------------------
  
  	/*for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%d ;",A[i]);
	}
	printf("\n---------\n");
	
	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%d ;",B[i]);
	}
	printf("\n---------\n");
	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%d ;",C[i]);
	}
	printf("\n---------\n");*/


	free(A);
	free(B);
	free(C1);
	free(C2);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	
	return 0;
}