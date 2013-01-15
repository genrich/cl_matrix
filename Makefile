CFLAGS=-c -std=c++11 -fPIC -g -O2
LDLAGS=-shared -Wl,-Bsymbolic
INC_DIRS=include $(AMDAPPSDKROOT)/include $(AMDAPPBLASSDKROOT)/include /usr/local/include/octave-3.6.3
LIB_DIRS=/usr/lib/fglrx $(AMDAPPBLASSDKROOT)/lib64 /usr/local/lib/octave/3.6.3 /usr/local/lib 
LIBS=OpenCL clAmdBlas octinterp octave cruft

CFLAGS_TEST=-std=c++11 -g
LIBS_TEST=$(LIBS) boost_unit_test_framework

.PHONY: all test macro

all: _build _build/octave_cl_matrix.oct cl_matrix.oct

_build:
	mkdir $@

_build/%.o: src/%.cpp
	$(CC) $(CFLAGS) $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS:%=-l%) -o $@ $^

_build/octave_cl_matrix.oct: _build/octave_cl_matrix.o _build/ClMatrix.o _build/ClAmdBlasService.o
	$(CC) $(LDLAGS) -o $@ $^ $(LIB_DIRS:%=-L%) $(LIBS:%=-l%)

cl_matrix.oct: _build/octave_cl_matrix.oct
	ln -s _build/octave_cl_matrix.oct cl_matrix.oct

_build/test: test/ClMatrixTest.cpp _build/ClMatrix.o _build/ClAmdBlasService.o
	$(CC) $(CFLAGS_TEST) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS_TEST:%=-l%)

clean:
	rm _build/*

test: _build/test
	$^ --log_level=message
