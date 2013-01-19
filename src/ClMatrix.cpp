#include <stdio.h>
#include <string.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include "ClMatrix.hpp"
#include "ClService.hpp"

#include <clAmdBlas.h>

using namespace std;

extern ClService clSrvc;

cl_mem createEmpty (int rows, int cols)
{
    cl_int err;
    cl_mem mem;
    mem = clCreateBuffer (clSrvc.ctx (), CL_MEM_READ_WRITE, rows * cols * sizeof (double), nullptr, &err);
    if (CL_SUCCESS != err)
        throw runtime_error {"Create memory buffer (" + to_string (err) + ")"};
    return mem;
}

ClMatrix::ClMatrix (const int r, const int c)
    :ClMatrix (r, c, nullptr)
{
}

ClMatrix::ClMatrix (const int r, const int c, const double* data)
    :rows (r), cols (c), mem (createEmpty (r, c))
{
    if (nullptr != data)
    {
        cl_int err = clEnqueueWriteBuffer (clSrvc.queue (), mem, CL_TRUE, 0, r * c * sizeof (double), data, 0, NULL, NULL);
        if (CL_SUCCESS != err)
            throw runtime_error {"Couldn't write to the cl buffer object (" + to_string (err) + ")"};
    }
}

ClMatrix::ClMatrix (ClMatrix&& other)
    :rows (other.rows), cols (other.cols), mem (other.mem)
{
    other.mem = nullptr;
}

ClMatrix::~ClMatrix ()
{
    if (nullptr != mem)
        clReleaseMemObject (mem);
}

void ClMatrix::copyTo (double* data) const
{
    cl_int err = clEnqueueReadBuffer (clSrvc.queue (), mem, CL_TRUE, 0, byteSize (), data, 0, NULL, NULL);
    if (CL_SUCCESS != err)
        throw runtime_error {"Couldn't read from the cl buffer object (" + to_string (err) + ")"};
}

size_t ClMatrix::byteSize () const
{
    return rows * cols * sizeof (double);
}

ClMatrix ClMatrix::mul (const ClMatrix& other) const
{
    if (cols != other.rows) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {rows, other.cols};

    std::vector<cl::Event> event (1);
    clAmdBlasStatus status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasNoTrans,
            rows, other.cols, cols, 1/*alpha*/,
            mem, 0/*offset*/, rows/*lda*/,
            other.mem, 0/*offset*/, other.rows/*ldb*/,
            0/*beta*/, result.mem, 0/*offset*/, rows/*ldc*/,
            1/*commandQueues*/, &clSrvc.queue (), 0/*waitList*/, NULL, &event[0] ());
    if (clAmdBlasSuccess != status)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}

ClMatrix ClMatrix::el_mul (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element multiplication"};

    ClMatrix result {rows, cols};

    clSrvc.el_mul.setArg (0, mem);
    clSrvc.el_mul.setArg (1, other.mem);
    clSrvc.el_mul.setArg (2, result.mem);
    clSrvc.queue.enqueueNDRangeKernel (clSrvc.el_mul, 0, rows * cols);

    return result;
}

ClMatrix ClMatrix::operator* (const ClMatrix& other) const
{
    return mul (other);
}

ClMatrix ClMatrix::sigmoid () const
{
    ClMatrix result {rows, cols};

    clSrvc.sigmoid.setArg (0, mem);
    clSrvc.sigmoid.setArg (1, result.mem);
    clSrvc.queue.enqueueNDRangeKernel (clSrvc.sigmoid, 0, rows * cols);

    return result;
}
