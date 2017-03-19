#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <stdlib.h>

using namespace cv;

__global__ void BGRtoGrayScale (unsigned char* d_Pin, unsigned char* d_Pout, int n, int m) {
	// Calculate the row # of the d_Pin and d_Pout element to process
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the column # of the d_Pin and d_Pout element to process
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread computes one element of d_Pout if in range
	if ((Row < m) && (Col < n)) {
		d_Pout[Row*n+Col] = d_Pin[(Row*n+Col)*3]*0.07 + d_Pin[(Row*n+Col)*3+1]*0.72 + d_Pin[(Row*n+Col)*3+2]*0.21;
	}
}

void GPU_process(unsigned char& h_dataRawImage, unsigned char& h_imageOutput, int n, int m, int size, int sizeGray, int count, Mat& grayImage) {
	cudaError_t error = cudaSuccess;
	clock_t startGPU, endGPU;
	double gpu_time;
	unsigned char *d_dataRawImage, *d_imageOutput;
	int blockSize;


	error = cudaMalloc((void**)&d_dataRawImage, size);
	if(error != cudaSuccess) {
		printf("Error in cudaMalloc for d_dataRawImage\n");
		// exit(-1);
		return -1;
	}

	error = cudaMalloc((void**)&d_imageOutput, sizeGray);
	if(error != cudaSuccess) {
		printf("Error in cudaMalloc for d_imageOutput\n");
		// exit(-1);
		return -1;
	}

	printf("GPU_process begin...\n");

	for (int i = 0; i < count; ++i)
	{
		startGPU = clock();
		error = cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
			printf("Error copying data from h_dataRawImage to d_dataRawImage \n");
			// exit(-1);
			return -1;
		}

		blockSize = 32;

		dim3 dimBlock(blockSize, blockSize);
		dim3 dimGrid(ceil(n/float(blockSize)), ceil(m/float(blockSize)));

		BGRtoGrayScale<<<dimGrid, dimBlock>>> (d_dataRawImage, d_imageOutput, n, m);

		// cudaDeviceSynchronize();

		error = cudaMemcpy(h_imageOutput, d_imageOutput, sizeGray, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("Error copying data from d_imageOutput to h_imageOutput \n");
			// exit(-1);
			return -1;
		}

		endGPU = clock();
	}

	gpu_time = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
	printf("GPU time: %.10f\n", gpu_time);

	grayImage.data = h_imageOutput;

	// Write image gotten by GPU process on disk
	imwrite("\nGPU_res_GrayImage.jpg", grayImage);

	cudaFree(d_dataRawImage);
	cudaFree(d_imageOutput);
}

int main(int argc, char **argv) {
	clock_t startCPU, endCPU;
	double cpu_time;
	char* imageName = argv[1];
	long conv = strtol(argv[2], NULL, 10);
	unsigned char *h_dataRawImage, *h_imageOutput;
	Size s;
	int n, m, size, sizeGray, count = conv;

	Mat image;
	// Read the image
	image = imread(imageName, CV_LOAD_IMAGE_COLOR);

	// If image doesn't exists, or not image data given
	if(argc != 3 || !image.data) {
		printf("No image data\n");
		return -1;
	}
	// Get image dimension
	s = image.size();
	n = s.width;
	m = s.height;

	size = sizeof(unsigned char)*n*m*image.channels();
	sizeGray = sizeof(unsigned char)*n*m;

	h_dataRawImage = (unsigned char*)malloc(size);
	h_imageOutput = (unsigned char*)malloc(sizeGray);

	h_dataRawImage = image.data;

	Mat grayImage;
	grayImage.create(m,n,CV_8UC1);

	GPU_process(h_dataRawImage, h_imageOutput, n, m, size, sizeGray, count, grayImage) 


	/***********************CPU**************************/


	printf("CPU process begin...\n");

	for (int i = 0; i < count; ++i)
	{
		startCPU = clock();

		// Convert the image from BGR to gray scale format whit OpenCV
		cvtColor(image, grayImage, CV_BGR2GRAY);

		endCPU = clock();
	}

	/*********************END CPU************************/

	// Write image gotten by CPU process on disk
	imwrite("CPU_res_GrayImage.jpg", grayImage);

	cpu_time = ((double) (endCPU - startCPU)) / CLOCKS_PER_SEC;
	printf("CPU time: %.10f\n", cpu_time);

	free(h_imageOutput);

	return 0;
}
