#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string>
#include <math_functions.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernelTiled(float *A, float *B, float *C, int N){
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * TILE_WIDTH + tx;
	int row = by * TILE_WIDTH + ty;
	float Pvalue = 0.0;
	
	for(int m = 0; m < (TILE_WIDTH + N - 1)/TILE_WIDTH; ++m){
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


__global__ void matrixMultGPU (float *A, float *B, float *C, int N){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	float ans;
	if(col < N && row < N){
		ans = 0.0;
		for(int k=0;k<N;k++)
			ans += A[row*N+k] * B[k*N+col];
		C[row*N+col] = ans;
	}
}

void matrixMultCPU(float *A, float *B, float *C, int N){
	float ans;
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			ans=0.0;
			for(int k=0;k<N;k++)
				ans += A[i*N+k]*B[k*N+j];
			C[i*N+j] = ans;
		}
	}
}

std::string testValues(float *A, float *B, int N){
	for(int i = 0; i < N; ++i)
		for(int j = 0; j < N; ++j)
			if(A[(i*N)+j]!=B[(i*N)+j]){
				return "Wrong";
			}
		return "Correct";
}

void printMatrix(float *A, int N){
	for(int i=0;i<N*N;i++){
		if(i%N == 0)
		printf("\n");
			printf("%f; ",A[i]);
	}
	printf("\n---------\n");
}


void serial(float *A, float *B, float *C, double &time, int N) {

	/*******************************HOST********************************/
	clock_t tic = clock();
	matrixMultCPU(A,B,C, N);
  	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************END HOST******************************/

}

void checkError(cudaError_t error, std::string type) {
	if(error != cudaSuccess){
		printf("Error in %s\n", type.c_str());
		exit(0);
	}
}

void cuda(float *A, float *B, float *C, double &time, float size, int N) {
	cudaError_t error = cudaSuccess;
	float *d_A, *d_B, *d_C;

	error = cudaMalloc((void**)&d_A,size);
	checkError(error, "cudaMalloc for d_A (cuda)");

	error = cudaMalloc((void**)&d_B,size);
	checkError(error, "cudaMalloc for d_B (cuda)");

	error = cudaMalloc((void**)&d_C,size);
	checkError(error, "cudaMalloc for d_C (cuda)");

	/*******************************GPU********************************/
	clock_t tic = clock();

	error = cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_A (cuda)");

	error = cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_B (cuda)");
		
	dim3 dimBlock(32,32,1);
	dim3 dimGrid(ceil(N/float(dimBlock.x)),ceil(N/float(dimBlock.y)),1);

	matrixMultGPU<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);
	cudaDeviceSynchronize();
	error = cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
	checkError(error, "cudaMemcpy for C (cuda)");

	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU END******************************/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void cuda_tiled(float *A, float *B, float *C, double &time, float size, int N) {
	cudaError_t error = cudaSuccess;
	float *d_A, *d_B, *d_C;

	error = cudaMalloc((void**)&d_A,size);
	checkError(error, "cudaMalloc for d_A (cuda with tiling)");

	error = cudaMalloc((void**)&d_B,size);
	checkError(error, "cudaMalloc for d_B (cuda with tiling)");

	error = cudaMalloc((void**)&d_C,size);
	checkError(error, "cudaMalloc for d_C (cuda with tiling)");

	/*******************************GPU TILED********************************/
	clock_t tic = clock();

	error = cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_A (cuda with tiling)");

	error = cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_B (cuda with tiling)");
	
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	dim3 dimGrid(ceil(N/float(dimBlock.x)),ceil(N/float(dimBlock.y)),1);

	matrixMulKernelTiled<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);
	cudaDeviceSynchronize();
	error = cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
	checkError(error, "cudaMemcpy for C (cuda with tiling)");

	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU TILED END******************************/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


int main(int argc, char **argv){	
	float *A, *B, *C1, *C2, *C3;
	double CPU, GPU, GPU_tiled, acc1, acc2;
	CPU = GPU = GPU_tiled = acc1 = acc2 = 0.0;
	
	// Meaning of  positions: {serial, cuda, cuda+tiling}
	bool op[] = {false, false, false};

	if(argc < 2) {
		printf("No size given\n");
		return -1;
	}
	int N = atoi(argv[1]);

	for (int i = 2; i < argc; i++) {
		std::string s = argv[i];
		if (s == "s")
			op[0] = true;
		else if (s == "c")
			op[1] = true;
		else if (s == "ct")
			op[2] = true;
	}

	float size = N*N*sizeof(float);

  	A = (float*)malloc(size);
 	B = (float*)malloc(size);
 	C1 = (float*)malloc(size);
 	C2 = (float*)malloc(size);
	C3 = (float*)malloc(size);

	for(int i=0;i<N*N;i++){
			A[i]=1;
			B[i]=2;
	}

	if (op[0]) serial(A, B, C1, CPU, N);
	if (op[1]) cuda(A, B, C2, GPU, size, N);
	if (op[2]) cuda_tiled(A, B, C3, GPU_tiled, size, N);

	if (op[0]) {
		printf(" %f |", CPU);
	}
	else printf(" - |");

	if (op[1]) {
		if (op[0]) {
			acc1 = CPU / GPU;
			std::string r1 = testValues(C1, C2, N);
			printf(" %f | %f | %s |", GPU, acc1, r1.c_str());
		}
		else printf(" %f | - | - |", GPU);
	}
	else printf(" - | - | - |");

	if (op[2]) {
		if (op[1]) {
			acc2 = GPU / GPU_tiled;
			std::string r1 = testValues(C2, C3, N);
			printf(" %f | %f | %s |\n", GPU_tiled, acc2, r1.c_str());
		}
		else if (op[0]) {
			acc1 = CPU / GPU_tiled;
			std::string r1 = testValues(C1, C3, N);
			printf(" %f | %f | %s |\n", GPU_tiled, acc1, r1.c_str());
		}
		else printf(" %f | - | - |\n", GPU_tiled);
	}
	else printf(" - | - | - |\n");

	free(A);
	free(B);
	free(C1);
	free(C2);
	free(C3);
	
	return 0;
}
