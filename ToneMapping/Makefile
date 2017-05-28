CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tmo: tmo.cu
	$(CUDAC) $(LIBS) -o tmo tmo.cu $(CFLAGS)
clean:
	rm tmo