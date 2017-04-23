#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>
#include "opencv2/gpu/gpu.hpp"

#define MASK_SIZE 9
#define MASK_WIDTH 3
#define TILE_WIDTH 32

using namespace cv;

// Convolution matrix on constant memory
__constant__ char d_M[MASK_SIZE];


// Parallel Code on GPU using Constant Mem for matrix convol (CUDA)
__global__
void imgConvGPU(unsigned char* imgIn, int row, int col, /*unsigned int maskWidth,*/ unsigned char* imgOut) {
	unsigned int row_d = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col_d = blockIdx.x*blockDim.x+threadIdx.x;

	int start_r = row_d - (MASK_WIDTH/2);
	int start_c = col_d - (MASK_WIDTH/2);

	int Pixel = 0;

	for (int k = 0; k < MASK_WIDTH; ++k)
	{
		for (int l = 0; l < MASK_WIDTH; ++l)
		{
			if((k + start_r) >= 0 && (k + start_r) < row && (l + start_c) >= 0 && (l + start_c) < col)
				Pixel += imgIn[(k + start_r) * col + (l + start_c)] * d_M[k * MASK_WIDTH + l];
		}
	}

	Pixel = Pixel < 0 ? 0 : Pixel > 255 ? 255 : Pixel;
	imgOut[row_d * col + col_d] = (unsigned char)Pixel;
}

// Parallel Code on GPU using shared Mem (CUDA)
__global__
void imgConvGPU_sharedMem(unsigned char* imgIn, int row, int col, /*unsigned int maskWidth,*/ unsigned char* imgOut) {

	int dest, destX, destY, src, srcX, srcY, size_T = TILE_WIDTH + TILE_WIDTH - 1;

	unsigned int row_d = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col_d = blockIdx.x*blockDim.x+threadIdx.x;


	__shared__ char d_T[size_T][size_T];

	int n = MASK_WIDTH/2;

	dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
	destY = dest / size_T;
	destX = dest % size_T;
	srcY = blockIdx.y * TILE_WIDTH + destY - n;
	srcX = blockIdx.x * TILE_WIDTH + destX - n;
	src = srcY * col + srcX;
	if (srcY >= 0 && srcY < row && srcX >= 0 && srcX < col)
		d_T[destY][destX] = imgIn[src];
	else
		d_T[destY][destX] = 0;

	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	destY = dest / size_T;
	destX = dest % size_T;
	srcY = blockIdx.y * TILE_WIDTH + destY - n;
	srcX = blockIdx.x * TILE_WIDTH + destX - n;
	src = srcY * col + srcX;

	if (destY < size_T)
		if (srcY >= 0 && srcY < row && srcX >= 0 && srcX < col)
			d_T[destY][destX] = imgIn[src];
		else
			d_T[destY][destX] = 0;

	__syncthreads();

	int Pixel = 0;

	for (int k = 0; k < MASK_WIDTH; ++k)
	{
		for (int l = 0; l < MASK_WIDTH; ++l)
		{
			Pixel += d_T[threadIdx.y + k][threadIdx.x + l] * d_M[k * MASK_WIDTH + l];
		}
	}

	Pixel = Pixel < 0 ? 0 : Pixel > 255 ? 255 : Pixel;
	imgOut[row_d * col + col_d] = (unsigned char)Pixel;
}

void checkError(cudaError_t error, std::string type) {
	if(error != cudaSuccess){
		printf("Error in %s\n", type.c_str());
		exit(0);
	}
}


void cuda_const(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M, int size, double& time) {
	int size_M = sizeof(unsigned char)*MASK_SIZE;
	cudaError_t error = cudaSuccess;
	unsigned char *d_dataRawImage, *d_imageOutput;

	error = cudaMalloc((void**)&d_dataRawImage,size);
	checkError(error, "cudaMalloc for d_dataRawImage (cuda)");

	error = cudaMalloc((void**)&d_imageOutput,size);
	checkError(error, "cudaMalloc for d_imageOutput (cuda)");

	/*******************************GPU********************************/
	clock_t tic = clock();

	error = cudaMemcpy(d_dataRawImage,imgIn,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_dataRawImage (cuda)");

	error = cudaMemcpyToSymbol(d_M, M, size_M);
	checkError(error, "cudaMemcpyToSymbol for d_M (cuda)");

	dim3 dimBlock(32,32);
	dim3 dimGrid(ceil(col/float(dimBlock.x)),ceil(row/float(dimBlock.y)));

	imgConvGPU<<<dimGrid,dimBlock>>>(d_dataRawImage, row, col, maskWidth, d_imageOutput);
	cudaDeviceSynchronize();

	error = cudaMemcpy(imgOut,d_imageOutput,size,cudaMemcpyDeviceToHost);
	checkError(error, "cudaMemcpy for imgOut (cuda)");

	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU END******************************/

	cudaFree(d_dataRawImage);
	cudaFree(d_imageOutput);
}

void cuda_sm(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M, int size, double& time) {
	int size_M = sizeof(unsigned char)*MASK_SIZE;
	cudaError_t error = cudaSuccess;
	unsigned char *d_dataRawImage, *d_imageOutput;

	error = cudaMalloc((void**)&d_dataRawImage,size);
	checkError(error, "cudaMalloc for d_dataRawImage (cuda)");

	error = cudaMalloc((void**)&d_imageOutput,size);
	checkError(error, "cudaMalloc for d_imageOutput (cuda)");

	/*******************************GPU********************************/
	clock_t tic = clock();

	error = cudaMemcpy(d_dataRawImage,imgIn,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_dataRawImage (cuda)");

	error = cudaMemcpyToSymbol(d_M, M, size_M);
	checkError(error, "cudaMemcpyToSymbol for d_M (cuda)");

	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
	dim3 dimGrid(ceil(col/float(dimBlock.x)),ceil(row/float(dimBlock.y)));

	imgConvGPU_sharedMem<<<dimGrid,dimBlock>>>(d_dataRawImage, row, col, maskWidth, d_imageOutput);
	cudaDeviceSynchronize();

	error = cudaMemcpy(imgOut,d_imageOutput,size,cudaMemcpyDeviceToHost);
	checkError(error, "cudaMemcpy for imgOut (cuda)");

	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU END******************************/

	cudaFree(d_dataRawImage);
	cudaFree(d_imageOutput);
}

int main(int argc, char** argv)
{
	char M[] = {-1,0,1,-2,0,2,-1,0,1};
	unsigned int maskWidth = 3;

	/*
	imgIn: 		Original img (Gray scaled)
	imgOut_1:	Parallel w/ consntant mem
	imgOut_2:	Parallel w/ constant and shared mem
	imgOut_3:	
	imgOut_4:	
	*/

	unsigned char *imgIn, *imgOut_1, *imgOut_2, *imgOut_3;
	double GPU_C, GPU_CS, GPU, GPU_CV, acc1, acc2, acc3;
	GPU_C = GPU_CS = GPU = GPU_CV = acc1 = acc2 = acc3 = 0.0;

	// Meaning of  positions: {GPU_C, GPU_CS, GPU, GPU_CV}
	bool op[] = {false, false, false, false};

	if(argc < 2) {
		printf("No image name given\n");
		return -1;
	}
	char* imageName = argv[1];

	for (int i = 2; i < argc; i++) {
		std::string s = argv[i];
		if (s == "cconst")
			op[0] = true;
		else if (s == "csha")
			op[1] = true;
		else if (s == "pd")
			op[2] = true;
		else if (s == "sobel_d")
			op[3] = true;
	}

	Mat image;
	image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

	// Get image dimension
	Size s = image.size();
	int col = s.width;
	int row = s.height;

	int size = sizeof(unsigned char)*row*col;
	int sizeGray = sizeof(unsigned char)*row*col;

	imgIn = (unsigned char*)malloc(size);
	imgOut_1 = (unsigned char*)malloc(sizeGray);
	imgOut_3 = (unsigned char*)malloc(sizeGray);

	imgIn = image.data;

	Mat result, imgOut_4;
	imgOut_4.create(row,col,CV_8UC1);

	if (op[0]) cuda_const(imgIn, row, col, maskWidth, imgOut_1, M, GPU_C);
	if (op[1]) cuda_sm(imgIn, row, col, maskWidth, imgOut_1, M, GPU_CS);
	// if (op[2]) parallel_device(imgIn, row, col, maskWidth, imgOut_3, M, sizeGray, GPU);
	// if (op[3]) sobel_device(image, imgOut_4, GPU_CV);

	result.create(row,col,CV_8UC1);

	if (op[0]) {
		printf(" %f |", GPU_C);
		result.data = imgOut_1;
		imwrite("res_GPU_C.jpg", result);
	}
	else printf(" - |");

	if (op[1]) {
		if (op[0]) {
			acc1 = GPU_C / GPU_CS;
			printf(" %f | %f |", GPU_CS, acc1);
		}
		else printf(" %f | - |", GPU_CS);
		imwrite("res_GPU_CS.jpg", imgOut_2);
	}
	else printf(" - | - |");

	if (op[2]) {
		if (op[0]) {
			acc2 = GPU_C / GPU;
			printf(" %f | %f |", GPU, acc2);
		}
		else printf(" %f | - |", GPU);
		result.data = imgOut_3;
		imwrite("res_GPU.jpg", result);
	}
	else printf(" - | - |");

	if (op[3]) {
		if (op[0]) {
			acc3 = GPU_C / GPU_CV;
			printf(" %f | %f |\n", GPU_CV, acc3);
		}
		else printf(" %f | - |\n", GPU_CV);
		imwrite("res_GPU_CV.jpg", imgOut_4);
	}
	else printf(" - | - |\n");


	free(imgOut_1);
	free(imgOut_3);

	return 0;
}
