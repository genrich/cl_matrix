CFLAGS=-c -std=c++11 -fPIC -g -O2
LDLAGS=-shared -Wl,-Bsymbolic
INC_DIRS=include $(AMDAPPSDKROOT)/include $(AMDAPPBLASSDKROOT)/include /usr/local/include/octave-3.6.3
LIB_DIRS=/usr/lib/fglrx $(AMDAPPBLASSDKROOT)/lib64 /usr/local/lib/octave/3.6.3 /usr/local/lib 
LIBS=OpenCL clAmdBlas octinterp octave cruft

CFLAGS_TEST=-std=c++11 -g
LIBS_TEST=$(LIBS) boost_unit_test_framework

.PHONY: all test clean

all: _build cl_matrix.oct kernels

_build:
	mkdir $@

_build/%.o: src/%.cpp
	$(CC) $(CFLAGS) $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS:%=-l%) -o $@ $^

cl_matrix.oct: _build/octave_cl_matrix.o _build/ClMatrix.o _build/ClService.o
	$(CC) $(LDLAGS) -o $@ $^ $(LIB_DIRS:%=-L%) $(LIBS:%=-l%)

_build/compile_kernels: src/compile_kernels.cpp src/ClService.cpp
	$(CC) $(CFLAGS_TEST) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS:%=-l%)

kernels: _build/compile_kernels src/kernels.cl
	_build/compile_kernels

_build/test: test/ClMatrixTest.cpp _build/ClMatrix.o _build/ClService.o
	$(CC) $(CFLAGS_TEST) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS_TEST:%=-l%)

test: _build/test kernels
	_build/test --log_level=message

clean:
	rm -f cl_matrix.oct kernels _build/*
