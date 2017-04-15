CUDAC 	= nvcc
CFLAGS	= `pkg-config --cflags opencv`
LIBS	= `pkg-config --libs opencv`

all: ImageConvolution

ImageConvolution:	ImageConvolution.cu
	$(CUDAC) $(CFLAGS) ImageConvolution.cu $(LIBS) -o ImageConvolution

clean:
	rm -f ImageConvolution