ICPC	:= /opt/intel/bin/icpc
NVCC	:= /usr/local/cuda-5.0/bin/nvcc
ARCH	:= sm_35

INCLUDE := -I ../include \
		   -I .
CUINCLUDE:= $(INCLUDE) -I/usr/local/cuda-5.0/samples/common/inc/ -I/usr/local/cuda-5.0/include/

CPPFLAGS := -fopenmp $(INCLUDE) -L/opt/intel/composer_xe_2013.2.146/compiler/lib/intel64 -L/opt/intel/composer_xe_2013.2.146/compiler/lib/mic -w
MICFLAGS := -mmic -fopenmp -vec-report1 -openmp-report1 $(INCLUDE) -L/opt/intel/composer_xe_2013.2.146/compiler/lib/mic -liomp5 -w
CUFLAGS := -O2 -arch $(ARCH) $(CUINCLUDE) -L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib -w 

CUFILES := $(notdir $(wildcard *.cu))
CPPFILES := $(notdir $(wildcard *.cpp))

CUBINS  := $(patsubst %.cu,%.out,$(CUFILES))
CPPBINS := $(patsubst %.cpp,%.bin,$(CPPFILES))

all: $(CUBINS) $(CPPBINS) Makefile

clean :
	rm -rf *.bin
	rm -rf *.out
	rm -rf *.S
	rm -rf *.mic

$(CUBINS): $(CUFILES)
	$(NVCC) $(CUFLAGS) $(patsubst %.out,%.cu,$@) -o $@

$(CPPBINS): $(CPPFILES)
	$(ICPC) $(CPPFLAGS) $(patsubst %.bin,%.cpp,$@) -o $@
	$(ICPC) $(MICFLAGS) $(patsubst %.bin,%.cpp,$@) -o $(patsubst %.bin,%.mic,$@)
	$(ICPC) -S $(CPPFLAGS) $(patsubst %.bin,%.cpp,$@) -o $(patsubst %.bin,%.S,$@)

