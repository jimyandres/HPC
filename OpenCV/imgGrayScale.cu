#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

using namespace cv;

__global__ void PictureKernell (unsigned char* d_Pin, unsigned char* d_Pout, int n, int m) {
	// Calculate the row # of the d_Pin and d_Pout element to process
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the column # of the d_Pin and d_Pout element to process
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread computes one element of d_Pout if in range
	if ((Row < m) && (Col < n)) {
		d_Pout[Row*n+Col] = 2*d_Pin[Row*n+Col];
	}
}

int main(int argc, char **argv) {
	cudaError_t error = cudaSuccess;
	clock_t startGPU, endGPU;
	double gpu_time;
	char* imageName = argv[1];
	unsigned char *h_dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
	Size s;
	int n, m, size, sizeGray, blockSize;

	Mat image;
	// Read the image on gray scale
	image = imread(imageName,CV_LOAD_IMAGE_GRAYSCALE);

	// If image doesn't exists, or no image data given
	if(argc != 2 || !image.data) {
		printf("No image data\n");
		return -1;
	}

	s = image.size();
	n = s.width;
	m = s.height;

	size = sizeof(unsigned char)*n*m;
	sizeGray = sizeof(unsigned char)*n*m;

	h_dataRawImage = (unsigned char*)malloc(size);
	error = cudaMalloc((void**))&d_dataRawImage, size);
	if(error != cudaSuccess) {
		printf("Error in cudaMalloc for d_dataRawImage\n");
		// exit(-1);
		return -1;
	}

	h_imageOutput = (unsigned char*)malloc(sizeGray);
	error = cudaMalloc((void**)&d_imageOutput, sizeGray);
	if(error != cudaSuccess) {
		printf("Error in cudaMalloc for d_imageOutput\n");
		// exit(-1);
		return -1;
	}

	h_dataRawImage = image.data;

	startGPU = clock();
	error = cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Error copying data from h_dataRawImage to d_dataRawImage \n");
		// exit(-1);
		return -1;
	}

	blockSize = 32;

	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(n/float(blockSize)), ceil(m/float(blockSize)), 1);

	PictureKernell<<<dimGrid, dimBlock>>> (d_dataRawImage, d_imageOutput, n, m);

	// cudaDeviceSynchronize();

	error = cudaMemcpy(h_imageOutput, d_imageOutput, sizeGray, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		printf("Error copying data from d_imageOutput to h_imageOutput \n");
		// exit(-1);
		return -1;
	}

	endGPU = clock();
	

	Mat grayImage;

	grayImage.create(n,m,CV_8UC1);
	grayImage.data = h_imageOutput;

	// Convert the image from BGR to gray scale format
	// cvtColor(image, grayImage, CV_BGR2GRAY);

	// Write image on disk
	imwrite("res_GrayImage.jpg", grayImage);

	gpu_time = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
	printf("GPU time: %.10f\n", gpu_time);

	free(h_dataRawImage);
	free(h_imageOutput);
	cudaFree(d_dataRawImage);
	cudaFree(d_imageOutput);

	return 0;
}