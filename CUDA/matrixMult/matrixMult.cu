#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


__global__ void matrixMultGPU (float *A, float *B, float *C, long N){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	float acc;
	if(col < N && row < N){
		acc = 0.0;
		for(int k=0;k<N;k++)
			acc += A[row*N+k] * B[k*N+col];
		C[row*N+col] = acc;
	}
}

void matrixMultCPU(float *A, float *B, float *C, long N){
	float acc;
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			acc=0.0;
			for(int k=0;k<N;k++){
				acc += A[i*N+k]*B[k*N+j];
			}
		C[i*N+j] = acc;
		}
	}
}

int main(int argc, char **argv){
	
	float *A, *B, *C1, *C2;
	float *d_A, *d_B, *d_C;
	double CPU, GPU;
	if(argc != 2) {
		printf("No size given\n");
		return -1;
	}
	long N = strtol(argv[1], NULL, 10);

	float size = N*N*sizeof(float);

  	A = (float*)malloc(size);
 	B = (float*)malloc(size);
 	C1 = (float*)malloc(size);
 	C2 = (float*)malloc(size);

	for(int i=0;i<N*N;i++){
			A[i]=1.0;
			B[i]=2.0;
	}

	//CPU----------------------------
	clock_t tic = clock();
	matrixMultCPU(A,B,C1, N);
  	clock_t toc = clock();
	//printf("Tiempo CPU: %f segundos", (double)(toc - tic) / CLOCKS_PER_SEC);
	CPU = (double)(toc - tic) / CLOCKS_PER_SEC;
	printf("%f,",CPU);
	//-------------------------------
  
	cudaMalloc((void**)&d_A,size);
	cudaMalloc((void**)&d_B,size);
	cudaMalloc((void**)&d_C,size);

	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

	//GPU----------------------------
	dim3 dimBlock(32,32);
  	dim3 dimGrid(ceil(N/float(dimBlock.x)),ceil(N/float(dimBlock.y)),1);
	
  	clock_t tic2 = clock();
	matrixMultGPU<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);
  	//cudaDeviceSynchronize();
	cudaMemcpy(C2,d_C,size,cudaMemcpyDeviceToHost);
  	clock_t toc2 = clock();
	//printf("\n\nTiempo GPU: %f segundos\n", (double)(toc2 - tic2) / CLOCKS_PER_SEC);
	GPU = (double)(toc2 - tic2) / CLOCKS_PER_SEC;
	printf("%f,%f\n",GPU,(CPU/GPU));
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
