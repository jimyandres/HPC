CUDAC 	= nvcc

all: matrixMult

matrixMult:	matrixMult.cu
	$(CUDAC) matrixMult.cu -o matrixMult

clean:
	rm -f matrixMult