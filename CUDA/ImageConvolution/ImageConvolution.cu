#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define RED 2
#define GREEN 1
#define BLUE 0

// __constant__ char M[MASK_WIDTH*MASK_WIDTH];

// Sequential Code on CPU
void imgConvCPU(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M) {

	int start_M = maskWidth/2;

	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			int Pixel = 0;
			for (int k = 0; k < maskWidth; ++k)
			{
				for (int l = 0; l < maskWidth; ++l)
				{
					if((k - start_M) > 0 && (k - start_M) < row && (l - start_M) > 0 && (l - start_M) > col)
						Pixel += imgIn[(k - start_M) * row + (l - start_M)] * M[k * maskWidth + l];
				}
			}
			Pixel = Pixel < 0 ? 0 : Pixel;
			Pixel = Pixel > 255 ? 255 : Pixel;
			imgOut[i * row + j] = (unsigned char)Pixel;
		}
	}

}

void serial_host(unsigned char* imgIn, int row, int col, unsigned int maskWidth, unsigned char* imgOut, char* M, double &time) {
	/*******************************HOST********************************/
	clock_t tic = clock();
	matrixMultCPU(A,B,C, N);
  	clock_t toc = clock();
	time = (double)(toc - tic) / CLOCKS_PER_SEC;
	/*****************************END HOST******************************/
}

int main(int argc, char const *argv[])
{
	char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
	unsigned int maskWidth = 3;
	/*
	imgIn: 		Original img (Gray scaled)
	imgOut_1:	Sequential convolution on host
	imgOut_2:	Sobel on host
	imgOut_3:	Sequential convolution on device
	imgOut_4:	Sobel on device
	*/
	unsigned char *imgIn, *imgOut_1, *imgOut_2, *imgOut_3, *imgOut_4;
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
			op[1] = true;
		else if (s == "sobel_d")
			op[2] = true;
	}

	Mat image;
	image = imread(imageName, 1);


	// Get image dimension
	Size s = image.size();
	int col = s.width;
	int row = s.height;
	
	int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;

    imgIn = (unsigned char*)malloc(size);
	imgOut_1 = (unsigned char*)malloc(sizeGray);
	imgOut_2 = (unsigned char*)malloc(sizeGray);
	imgOut_3 = (unsigned char*)malloc(sizeGray);
	imgOut_4 = (unsigned char*)malloc(sizeGray);

	imgIn = image.data;	

	if (op[0]) serial_host(imgIn, row, col, maskWidth, imgOut_1, M, CPU);
	// if (op[1]) sobel_host(A, B, C2, GPU, size, N);
	// if (op[2]) serial_device(A, B, C3, GPU_tiled, size, N);
	// if (op[3]) sobel_device(A, B, C3, GPU_tiled, size, N);

	if (op[0]) {
		printf(" %f |", CPU);
	}
	else printf(" - |");

	if (op[1]) {
		f (op[0]) {
			acc1 = CPU / CPU_CV;
			printf(" %f | %f |", CPU_CV, acc1);
		}
		else printf(" %f | - |", CPU_CV);
	}
	else printf(" - | - |");

	if (op[2]) {
		f (op[0]) {
			acc2 = CPU / GPU;
			printf(" %f | %f |", GPU, acc2);
		}
		else printf(" %f | - |", GPU);
	}
	else printf(" - | - |");

	if (op[3]) {
		f (op[0]) {
			acc3 = CPU / GPU_CV;
			printf(" %f | %f |", GPU_CV, acc3);
		}
		else printf(" %f | - |", GPU_CV);
	}
	else printf(" - | - |");


	free(imgIn);
	free(imgOut_1);
	free(imgOut_2);
	free(imgOut_3);
	free(imgOut_4);
	
	return 0;
}