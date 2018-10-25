# cbert/Makefile

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math -lcurand -Xcompiler -Wall

all: main shared
	
main: src/cbert.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/cbert src/cbert.cu -I src

shared: src/cbert.cu
	mkdir -p lib
	nvcc $(ARCH) $(OPTIONS) --std=c++11 -Xcompiler -fPIC -shared -o lib/cbert.so src/cbert.cu -I src
	
clean:
	rm -rf bin lib