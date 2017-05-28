#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void checkError(cudaError_t err) {
    if(err!=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

//Convert a pixel RGB to HSL (Hue Saturation Lightness)
__device__ float4 convert_pixel_to_hsl(float4 pixel) {
    float r, g, b, a;
    float h, s, l, v;

    r = pixel.x / 255.0f;
    g = pixel.y / 255.0f;
    b = pixel.z / 255.0f;
    a = pixel.w;

    float max = fmax(r, fmax(g, b));
    float min = fmin(r, fmin(g, b));
    float diff = max - min;

    v = max;
    l = diff/2;

    if(v == 0.0f) // black
        h = s = 0.0f;
    else {
        s = diff / v;
        if(diff < 0.001f)  // grey
            h = 0.0f;
        else { // color
            if(max == r) {
                h = 60.0f * (g - b)/diff;
                if(h < 0.0f)
                    h += 360.0f;
            } else if(max == g)
                h = 60.0f * (2 + (b - r)/diff);
            else
                h = 60.0f * (4 + (r - g)/diff);
        }
    }
    return (float4) {h, s, l, a};
}

__device__ float4 convert_pixel_to_rgb(float4 pixel) {
    float r, g, b;
    float h, s, l;
    float c, x, hi, m;

    h = pixel.x;
    s = pixel.y;
    l = pixel.z;

    hi = h/60.0f;

    c = (1 - fabsf(2*l -1)) * s;
    x = c * (1 - fabsf(fmodf(hi, 2) - 1));
    m = (l - c)/2;

    if(h >= 0.0f && h < 60.0f) {
        r = c;
        g = x;
        b = 0;
    } else if(h >= 60.0f && h < 120.0f) {
        r = x;
        g = c;
        b = 0;
    } else if(h >= 120.0f && h < 180.0f) {
        r = 0;
        g = c;
        b = x;
    } else if(h >= 180.0f && h < 240.0f) {
        r = 0;
        g = x;
        b = c;
    } else if(h >= 240.0f && h < 300.0f) {
        r = x;
        g = 0;
        b = c;
    } else if(h >= 300.0f && h < 360.0f) {
        r = c;
        g = 0;
        b = x;
    }

    r = (r + m)*255.0f;
    g = (g + m)*255.0f;
    b = (b + m)*255.0f;

    return (float4) {r, g, b, pixel.w};
}

__device__ float log_mapping(float world_lum, float max_lum, float q, float k, float lum) {
    float a, b, lum_d;
    a = 1 + q * world_lum;
    b = 1 + k * max_lum;
    lum_d = log10f(a)/log10f(b);

    return lum_d;
}

__global__ void tmo(float* imageIn, float* imageOut, int width, int height, float world_lum,
                    float max_lum, float q, float k)
{
    float p_red, p_green, p_blue;
    float4 pixel_hsl, pixel_rgb;

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if(Row < height && Col < width) {
        p_red = imageIn[(Row*width+Col)*3+RED];
        p_green = imageIn[(Row*width+Col)*3+GREEN];
        p_blue = imageIn[(Row*width+Col)*3+BLUE];

        pixel_hsl = convert_pixel_to_hsl(make_float4(p_red, p_green, p_blue, 0.0));

        pixel_hsl.z = log_mapping(world_lum, max_lum, q, k, pixel_hsl.z);

        pixel_rgb = convert_pixel_to_rgb(pixel_hsl);

        imageOut[(Row*width+Col)*3+BLUE] = pixel_rgb.z;
        imageOut[(Row*width+Col)*3+GREEN] = pixel_rgb.y;
        imageOut[(Row*width+Col)*3+RED] = pixel_rgb.x;
    }
}

// Iout(x,y)=(Iin(x,y)⋅2ᶺf)ᶺ(1/g)

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
    return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__global__ void tonemap(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float f_stop,
                        float gamma)
{
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if(Row < height && Col < width) {
        imageOut[(Row*width+Col)*3+BLUE] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+BLUE]);
        imageOut[(Row*width+Col)*3+GREEN] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+GREEN]);
        imageOut[(Row*width+Col)*3+RED] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+RED]);
    }
}

void showImage(Mat &image, const char *window) {
    namedWindow(window, CV_WINDOW_NORMAL);
    imshow(window, image);
}

int main(int argc, char** argv)
{
    char* image_name = argv[1];
    char* image_out_name = argv[6];
    float *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut;
    Mat hdr, ldr;
    Size imageSize;
    int width, height, channels, sizeImage;
    float world_lum=0.0, max_lum=0.0, q=0.0, k=0.0;
    int show_flag = 0;
//	std::vector<Mat>images;

//	printf("%s\n", image_name);
    hdr = imread(image_name, -1);
    if(argc !=6 || !hdr.data) {
        printf("No image Data \n");
        printf("Usage: ./test <file_path> <world_lum> <max_lum> <q> <k> <image_out_name>");
        return -1;
    }

    world_lum = atof(argv[2]);
    max_lum = atof(argv[3]);
    q = atof(argv[4]);
    k = atof(argv[5]);

    if(hdr.empty()) {
        printf("Couldn't find or open the image...\n");
        return -1;
    }
    imageSize = hdr.size();
    width = imageSize.width;
    height = imageSize.height;
    channels = hdr.channels();
    sizeImage = sizeof(float)*width*height*channels;

    //printf("Width: %d\nHeight: %d\n", width, height);
    std::string ty =  type2str( hdr.type() );
//	printf("Image: %s %dx%d \n", ty.c_str(), hdr.cols, hdr.rows );

    //printf("Channels: %d\nDepth: %d\n", hdr.channels(), hdr.depth());

    h_ImageData = (float *) malloc (sizeImage);
    h_ImageData = (float *)hdr.data;
    h_ImageOut = (float *) malloc (sizeImage);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    checkError(cudaMalloc((void **)&d_ImageData, sizeImage));
    checkError(cudaMalloc((void **)&d_ImageOut, sizeImage));
    checkError(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
    cudaEventRecord(start);
//    tonemap<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, 32, f_stop, gamma);
    tmo<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, world_lum, max_lum, q, k);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%s|%.10f\n", image_name, milliseconds/1000.0);

    checkError(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

    ldr.create(height, width, CV_32FC3);
    ldr.data = (unsigned char *)h_ImageOut;
    ldr.convertTo(ldr, CV_8UC3, 255);
    imwrite(image_out_name, ldr);

    ty =  type2str( ldr.type() );
//    printf("Image result: %s %dx%d \n", ty.c_str(), ldr.cols, ldr.rows );

    if(show_flag) {
        showImage(ldr, "Image out LDR");
        waitKey(0);
    }

    free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut);

    return 0;
}
