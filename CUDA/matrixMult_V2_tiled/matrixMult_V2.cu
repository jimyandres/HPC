#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string>
#include <math_functions.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernelTiled(double *A, double *B, double *C, int N){
	__shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * TILE_WIDTH + tx;
	int row = by * TILE_WIDTH + ty;
	double Pvalue = 0.0;
	
	int Ntiles = (TILE_WIDTH + N - 1)/TILE_WIDTH;
	for(int m = 0; m < Ntiles; ++m){
		if ((m*TILE_WIDTH + tx) < N && row < N)
			Mds[ty][tx] = A[row*N + m*TILE_WIDTH + tx];
		else
			Mds[ty][tx] = 0.0;
		if ((m*TILE_WIDTH + ty) < N && col < N)
			Nds[ty][tx] = B[(m*TILE_WIDTH + ty) * N + col];
		else
			Nds[ty][tx] = 0.0;

		__syncthreads();

		for(int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}
	if (row < N && col < N)
		C[row*N+col] = Pvalue;
}


__global__ void matrixMultGPU (double *A, double *B, double *C, int N){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	double acc;
	if(col < N && row < N){
		acc = 0.0;
		for(int k=0;k<N;k++)
			acc += A[row*N+k] * B[k*N+col];
		C[row*N+col] = acc;
	}
}

void matrixMultCPU(double *A, double *B, double *C, int N){
	double acc;
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			acc=0.0;
			for(int k=0;k<N;k++)
				acc += A[i*N+k]*B[k*N+j];
			C[i*N+j] = acc;
		}
	}
}

std::string testValues(double *A, double *B, int N){
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            if(A[(i*N)+j]!=B[(i*N)+j]){
                return "Mal Cálculo";
            }
    return "Buen Cálculo";
}


int main(int argc, char **argv){
	cudaError_t error = cudaSuccess;	
	double *A, *B, *C1, *C2;
	double *d_A, *d_B, *d_C;
	double CPU, GPU_tiled;
	if(argc != 2) {
		printf("No size given\n");
		return -1;
	}
	int N = atoi(argv[1]);

	double size = N*N*sizeof(double);

  	A = (double*)malloc(size);
 	B = (double*)malloc(size);
 	C1 = (double*)malloc(size);
 	C2 = (double*)malloc(size);

	if (C2 == NULL)
		return 0;

	for(int i=0;i<N*N;i++){
			A[i]=1;
			B[i]=2;
	}

	/*******************************HOST********************************/
	clock_t tic = clock();
	matrixMultCPU(A,B,C1, N);
  	clock_t toc = clock();
	//printf("Tiempo CPU: %f segundos", (double)(toc - tic) / CLOCKS_PER_SEC);
	CPU = (double)(toc - tic) / CLOCKS_PER_SEC;
	printf("%f,",CPU);
	/*****************************END HOST******************************/
  
	error = cudaMalloc((void**)&d_A,size);
	if(error != cudaSuccess){
		printf("Error in cudaMalloc for d_A\n");
		exit(0);
	}
	
	error = cudaMalloc((void**)&d_B,size);
	if(error != cudaSuccess){
                printf("Error in cudaMalloc for d_B\n");
                exit(0);
        }

	error = cudaMalloc((void**)&d_C,size);
	if(error != cudaSuccess){
                printf("Error in cudaMalloc for d_C\n");
                exit(0);
        }

	error = cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
                printf("Error in cudaMemcpy for d_A\n");
                exit(0);
        }

	error = cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
                printf("Error in cudaMemcpy for d_B\n");
                exit(0);
        }

	/*******************************GPU TILED********************************/
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
	dim3 dimGrid(ceil(N/float(dimBlock.x)),ceil(N/float(dimBlock.y)),1);
	
  	clock_t tic2 = clock();
	matrixMulKernelTiled<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);
	//matrixMultGPU<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);

	cudaDeviceSynchronize();

	cudaMemcpy(C2,d_C,size,cudaMemcpyDeviceToHost);
  	clock_t toc2 = clock();
	//printf("\n\nTiempo GPU: %f segundos\n", (double)(toc2 - tic2) / CLOCKS_PER_SEC);
	GPU_tiled = (double)(toc2 - tic2) / CLOCKS_PER_SEC;
	printf("%f,%f,%s\n", GPU_tiled, (CPU/GPU_tiled), testValues(C1,C2,N).c_str());
	/*****************************GPU TILED END******************************/
  	/*
  	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%f ;",A[i]);
	}
	printf("\n---------\n");
	
	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%f ;",B[i]);
	}
	printf("\n---------\n");
	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%f ;",C1[i]);
	}
	printf("\n---------\n");
	for(int i=0;i<N*N;i++){
                if(i%N == 0)
                printf("\n");
                        printf("%f ;",C2[i]);
        }
        printf("\n---------\n");
	*/	


	free(A);
	free(B);
	free(C1);
	free(C2);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	
	return 0;
}
