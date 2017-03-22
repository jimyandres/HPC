CUDAC 	= nvcc

all: matrixMult

matrixMult:	matrixMult.cu
	$(CUDAC) matrixMult.cu -o matrixMult -O3

clean:
	rm -f matrixMult
