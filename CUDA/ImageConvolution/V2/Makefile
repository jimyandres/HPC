CUDAC 	= nvcc
CFLAGS	= `pkg-config --cflags opencv` -O3 -arch=compute_52
LIBS	= `pkg-config --libs opencv`

all: ImageConvolution_V2

ImageConvolution_V2:	ImageConvolution_V2.cu
	$(CUDAC) $(CFLAGS) ImageConvolution_V2.cu $(LIBS) -o ImageConvolution_V2 -

clean:
	rm -f ImageConvolution_V2
