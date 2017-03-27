CUDAC 	= nvcc

FLAGS 	= -O3 -arch=sm_30

all: matrixMult_V2

matrixMult_V2:	matrixMult_V2.cu
	$(CUDAC) matrixMult_V2.cu -o matrixMult_V2 $(FLAGS)

clean:
	rm -f matrixMult_V2
