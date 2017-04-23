CUDAC 	= nvcc
CFLAGS	= `pkg-config --cflags opencv`
LIBS	= `pkg-config --libs opencv`

all: ImageConvolution_V2

ImageConvolution_V2:	ImageConvolution_V2.cu
	$(CUDAC) $(CFLAGS) -lineinfo ImageConvolution_V2.cu $(LIBS) -o ImageConvolution_V2 -O3 -arch=compute_52

clean:
	rm -f ImageConvolution_V2
