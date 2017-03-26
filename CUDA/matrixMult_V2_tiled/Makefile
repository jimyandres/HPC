CUDAC 	= nvcc

all: matrixMult_V2

matrixMult_V2:	matrixMult_V2.cu
	$(CUDAC) matrixMult_V2.cu -o matrixMult_V2 -O3

clean:
	rm -f matrixMult_V2
