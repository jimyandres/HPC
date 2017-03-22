#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


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

int main(int argc, char **argv){
	cudaError_t error = cudaSuccess;	
	double *A, *B, *C1, *C2;
	double *d_A, *d_B, *d_C;
	double CPU, GPU;
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

	for(int i=0;i<N*N;i++){
			A[i]=1;
			B[i]=2;
	}

	//CPU----------------------------
	clock_t tic = clock();
	matrixMultCPU(A,B,C1, N);
  	clock_t toc = clock();
	//printf("Tiempo CPU: %f segundos", (double)(toc - tic) / CLOCKS_PER_SEC);
	CPU = (double)(toc - tic) / CLOCKS_PER_SEC;
	printf("%f,",CPU);
	//-------------------------------
  
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

	//GPU----------------------------
	dim3 dimBlock(32,32);
  	dim3 dimGrid(ceil(N/(dimBlock.x)),ceil(N/(dimBlock.y)));
	
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
