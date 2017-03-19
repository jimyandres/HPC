CUDAC 	= nvcc
CFLAGS	= `pkg-config --cflags opencv`
LIBS	= `pkg-config --libs opencv`

all: rgb_grayscale

rgb_grayscale:	rgb_grayscale.cu
	$(CUDAC) $(CFLAGS) rgb_grayscale.cu $(LIBS) -o rgb_grayscale

clean:
	rm -f rgb_grayscale