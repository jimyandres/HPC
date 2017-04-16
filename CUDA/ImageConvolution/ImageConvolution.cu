#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>
#include "opencv2/gpu/gpu.hpp"

using namespace cv;

// Sequential Code on GPU (CUDA)
__global__
void imgConvGPU(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M) {
	unsigned int row_d = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col_d = blockIdx.x*blockDim.x+threadIdx.x;

	int start_r = row_d - (maskWidth/2);
	int start_c = col_d - (maskWidth/2);

	int Pixel = 0;

	for (int k = 0; k < maskWidth; ++k)
	{
		for (int l = 0; l < maskWidth; ++l)
		{
			if((k + start_r) >= 0 && (k + start_r) < row && (l + start_c) >= 0 && (l + start_c) < col)
				Pixel += imgIn[(k + start_r) * col + (l + start_c)] * M[k * maskWidth + l];
		}
	}

	Pixel = Pixel < 0 ? 0 : Pixel > 255 ? 255 : Pixel;
	imgOut[row_d * col + col_d] = (unsigned char)Pixel;
}

// Sequential Code on CPU
void imgConvCPU(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M) {
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			int Pixel = 0;
			int start_r = i - (maskWidth/2);
			int start_c = j - (maskWidth/2);
			for (int k = 0; k < maskWidth; ++k)
			{
				for (int l = 0; l < maskWidth; ++l)
				{
					if((k + start_r) >= 0 && (k + start_r) < row && (l + start_c) >= 0 && (l + start_c) < col)
						Pixel += imgIn[(k + start_r) * col + (l + start_c)] * M[k * maskWidth + l];
				}
			}
			Pixel = Pixel < 0 ? 0 : Pixel > 255 ? 255 : Pixel;
			imgOut[i * col + j] = (unsigned char)Pixel;
		}
	}

}

void checkError(cudaError_t error, std::string type) {
	if(error != cudaSuccess){
		printf("Error in %s\n", type.c_str());
		exit(0);
	}
}

void serial_host(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M, double& time) {
	/*******************************HOST********************************/
	clock_t tic = clock();
	imgConvCPU(imgIn,row,col,maskWidth,imgOut,M);
  	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************END HOST******************************/
}

void sobel_host(Mat& imgIn, Mat& imgOut, double& time){
	/*******************************HOST********************************/
	clock_t tic = clock();
	Sobel(imgIn,imgOut,CV_8UC1,1,0);
  	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************END HOST******************************/
}

void serial_device(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M, int size, double& time) {
	int size_M = sizeof(unsigned char)*9;
	cudaError_t error = cudaSuccess;
	unsigned char *d_dataRawImage, *d_imageOutput;
	char* d_M;

	error = cudaMalloc((void**)&d_dataRawImage,size);
	checkError(error, "cudaMalloc for d_dataRawImage (cuda)");

	error = cudaMalloc((void**)&d_imageOutput,size);
	checkError(error, "cudaMalloc for d_imageOutput (cuda)");

	error = cudaMalloc((void**)&d_M,size_M);
	checkError(error, "cudaMalloc for d_M (cuda)");

	/*******************************GPU********************************/
	clock_t tic = clock();

	error = cudaMemcpy(d_dataRawImage,imgIn,size,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_dataRawImage (cuda)");

	error = cudaMemcpy(d_M,M,size_M,cudaMemcpyHostToDevice);
	checkError(error, "cudaMemcpy for d_M (cuda)");
		
	dim3 dimBlock(32,32);
	dim3 dimGrid(ceil(col/float(dimBlock.x)),ceil(row/float(dimBlock.y)));

	imgConvGPU<<<dimGrid,dimBlock>>>(d_dataRawImage, row, col, maskWidth, d_imageOutput, d_M);
	cudaDeviceSynchronize();

	error = cudaMemcpy(imgOut,d_imageOutput,size,cudaMemcpyDeviceToHost);
	checkError(error, "cudaMemcpy for imgOut (cuda)");

	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU END******************************/

	cudaFree(d_dataRawImage);
	cudaFree(d_imageOutput);
	cudaFree(d_M);
}

void sobel_device(Mat& imgIn, Mat& imgOut, double& time){
	/*******************************GPU********************************/
	clock_t tic = clock();
	//gpu::setDevice(0);	

	// Copy the input image from CPU to GPU memory
	gpu::GpuMat gpuInput = gpu::GpuMat(imgIn);

	// Create the output
	gpu::GpuMat gpuOutput;

	// gradient y direction
	gpu::Sobel(gpuInput, gpuOutput, CV_8UC1, 1, 0);
	
	gpuOutput.download(imgOut);

  	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************GPU END******************************/

	gpuInput.release();
	gpuOutput.release();
}

int main(int argc, char** argv)
{
	char M[] = {-1,0,1,-2,0,2,-1,0,1};
	unsigned int maskWidth = 3;

	/*
	imgIn: 		Original img (Gray scaled)
	imgOut_1:	Sequential convolution on host
	imgOut_2:	Sobel on host
	imgOut_3:	Sequential convolution on device
	imgOut_4:	Sobel on device
	*/

	unsigned char *imgIn, *imgOut_1, *imgOut_3;
	double CPU, CPU_CV, GPU, GPU_CV, acc1, acc2, acc3;
	CPU = CPU_CV = GPU = GPU_CV = acc1 = acc2 = acc3 = 0.0;
	
	// Meaning of  positions: {CPU, CPU_CV, GPU, GPU_CV}
	bool op[] = {false, false, false, false};

	if(argc < 2) {
		printf("No image name given\n");
		return -1;
	}
	char* imageName = argv[1];

	for (int i = 2; i < argc; i++) {
		std::string s = argv[i];
		if (s == "seq_h")
			op[0] = true;
		else if (s == "sobel_h")
			op[1] = true;
		else if (s == "seq_d")
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

	Mat result, imgOut_2, imgOut_4;
	imgOut_2.create(row,col,CV_8UC1);
	imgOut_4.create(row,col,CV_8UC1);

	if (op[0]) serial_host(imgIn, row, col, maskWidth, imgOut_1, M, CPU);
	if (op[1]) sobel_host(image, imgOut_2, CPU_CV);
	if (op[2]) serial_device(imgIn, row, col, maskWidth, imgOut_3, M, sizeGray, GPU);
	if (op[3]) sobel_device(image, imgOut_4, GPU_CV);
	
	result.create(row,col,CV_8UC1);

	if (op[0]) {
		printf(" %f |", CPU);
		result.data = imgOut_1;
		imwrite("res_CPU.jpg", result);
	}
	else printf(" - |");

	if (op[1]) {
		if (op[0]) {
			acc1 = CPU / CPU_CV;
			printf(" %f | %f |", CPU_CV, acc1);
		}
		else printf(" %f | - |", CPU_CV);
		imwrite("res_CPU_CV.jpg", imgOut_2);
	}
	else printf(" - | - |");

	if (op[2]) {
		if (op[0]) {
			acc2 = CPU / GPU;
			printf(" %f | %f |", GPU, acc2);
		}
		else printf(" %f | - |", GPU);
		result.data = imgOut_3;
		imwrite("res_GPU.jpg", result);
	}
	else printf(" - | - |");

	if (op[3]) {
		if (op[0]) {
			acc3 = CPU / GPU_CV;
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
