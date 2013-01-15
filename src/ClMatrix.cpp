#include <stdio.h>
#include <string.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include <clAmdBlas.h>

#include "ClMatrix.hpp"
#include "ClAmdBlasService.hpp"


using namespace std;

extern ClAmdBlasService clSrvc;

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

ClMatrix ClMatrix::operator* (const ClMatrix& other) const
{
    ClMatrix newMatrix{rows, other.cols};

    std::vector<cl::Event> event (1);
    clAmdBlasStatus status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasNoTrans,
            rows, other.cols, cols, 1/*alpha*/,
            mem, 0/*offset*/, rows/*lda*/,
            other.mem, 0/*offset*/, other.rows/*ldb*/,
            0/*beta*/, newMatrix.mem, 0/*offset*/, rows/*ldc*/,
            1/*commandQueues*/, &clSrvc.queue (), 0/*waitList*/, NULL, &event[0] ());
    if (clAmdBlasSuccess != status)
        throw runtime_error {clSrvc.errMsg (status)};

    // cl::Event::waitForEvents (event);

    return newMatrix;
}

void ClMatrix::copyTo (double* data) const
{
    cl_int err = clEnqueueReadBuffer (clSrvc.queue (), mem, CL_TRUE, 0, rows * cols * sizeof (double), data, 0, NULL, NULL);
    if (CL_SUCCESS != err)
        throw runtime_error {"Couldn't read from the cl buffer object (" + to_string (err) + ")"};
}
