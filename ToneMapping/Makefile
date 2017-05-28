CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: tmo

tmo: tmo.cu
	$(CUDAC) $(LIBS) -lineinfo -o tmo tmo.cu $(CFLAGS) -O3 -arch=compute_52
clean:
	rm -f tmo