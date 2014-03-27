FILES   := divisionflopsD_error.cu
TARGET  := divisionflopsD_error.bin

NVCC	:= /usr/local/cuda-5.0/bin/nvcc
ARCH	:= sm_20

FLAGS := -O2 -arch $(ARCH) -I. -I/usr/local/cuda-5.0/samples/common/inc/ -I/usr/local/cuda-5.0/include/ -L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib

all:
	$(NVCC) $(FLAGS) $(FILES) -o $(TARGET)

clean :
	rm -rf *.out
