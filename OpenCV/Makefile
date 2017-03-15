CUDAC 	= nvcc
CFLAGS	= `pkg-config --cflags opencv`
LIBS	= `pkg-config --libs opencv`

imgGrayScale:	imgGrayScale.cu
	$(CUDAC) $(CFLAGS) imgGrayScale.cu $(LIBS) -o imgGrayScale
