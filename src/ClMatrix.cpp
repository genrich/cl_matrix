#include <stdio.h>
#include <string.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <cmath>

#include "ClMatrix.hpp"
#include "ClService.hpp"

#include <clAmdBlas.h>

using namespace std;

extern ClService clSrvc;

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define __(x) do {const int val = (x); if (val) throw runtime_error {"\n" __FILE__ ":" STRINGIZE(__LINE__) ":1 ERROR: " + clSrvc.errMsg (val)};} while (0)

//__________________________________________________________________________________________________

static size_t sizeVectorized (const size_t size)
{
    return ceil ((double) size / (double) clSrvc.vectorSize);
}
//__________________________________________________________________________________________________

static cl_mem createBuffer (const size_t bytes)
{
    cl_int err;
    cl_mem mem = clCreateBuffer (clSrvc.ctx (), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS)
        throw runtime_error {"Create memory buffer (" + to_string (err) + ")"};
    return mem;
}
//__________________________________________________________________________________________________

ClMatrix::ClMatrix (const size_t r, const size_t c):
    ClMatrix {r, c, nullptr}
{
}
//__________________________________________________________________________________________________

ClMatrix::ClMatrix (const size_t r, const size_t c, const double* data):
    rows        {r},
    cols        {c},
    size        {rows * cols},
    sizeVec     {sizeVectorized (size)},
    byteSize    {size * sizeof (double)},
    byteSizeVec {sizeVec * clSrvc.vectorSize * sizeof (double)},
    mem         {createBuffer (byteSizeVec)}
{
    size_t freeGlobMem[2];
    __(clGetDeviceInfo (clSrvc.device (), CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                        sizeof (freeGlobMem), freeGlobMem, nullptr));

    if (freeGlobMem[0] < clSrvc.globMem10Percent)
        __(clFinish (clSrvc.queue ()));

    if (data != nullptr)
    {
        const cl_int err = clEnqueueWriteBuffer (clSrvc.queue (), mem, CL_TRUE, 0,
                                                 byteSize, data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
            throw runtime_error {"Couldn't write to the cl buffer object (" + to_string (err) + ")"};
    }
}

ClMatrix::ClMatrix (const int r, const int c, const double* data):
    ClMatrix {(assert (r > 0), static_cast<size_t> (r)),
              (assert (c > 0), static_cast<size_t> (c)),
              data}
{
}

ClMatrix::ClMatrix (ClMatrix&& other):
    rows        {other.rows},
    cols        {other.cols},
    size        {other.size},
    sizeVec     {other.sizeVec},
    byteSize    {other.byteSize},
    byteSizeVec {other.byteSizeVec},
    mem         {other.mem}
{
    other.mem   = nullptr;
}
//__________________________________________________________________________________________________

ClMatrix::~ClMatrix ()
{
    if (mem != nullptr)
        __(clReleaseMemObject (mem));
}
//__________________________________________________________________________________________________

void ClMatrix::copyTo (double* data) const
{
    __(clFinish (clSrvc.queue ()));

    const cl_int err = clEnqueueReadBuffer (clSrvc.queue (), mem, CL_TRUE, 0, byteSize,
                                            data, 0, nullptr, nullptr);

    if (err != CL_SUCCESS)
        throw runtime_error {"Couldn't read from the cl buffer object (" + to_string (err) + ")"};
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::uminus () const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.uminus ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::transpose () const
{
    ClMatrix result {cols, rows};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {rows, cols};

    cl_kernel kernel = clSrvc.transpose ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 2, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::add (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element addition"};

    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.add ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::add (const double scalar) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.add_scalar ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (scalar), &scalar));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::sub (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element subtraction"};

    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.sub ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::sub (const double scalar) const
{
    return add (-scalar);
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::subtrahend (const double minuend) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.scalar_sub ();

    __(clSetKernelArg (kernel, 0, sizeof (minuend), &minuend));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::mul (const ClMatrix& other) const
{
    if (cols != other.rows) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {rows, other.cols};

    cl_command_queue queue = clSrvc.queue ();

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasNoTrans,
                                   rows, other.cols, cols, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, rows/*ldc*/,
                                   1/*commandQueues*/, &queue, 0, nullptr, nullptr);

    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::mul (const double scalar) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.mul_scalar ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (scalar), &scalar));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::el_mul (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element multiplication"};

    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.el_mul ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::trans_mul (const ClMatrix& other) const
{
    if (rows != other.rows) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {cols, other.cols};

    cl_command_queue queue = clSrvc.queue ();

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasTrans, clAmdBlasNoTrans,
                                   cols, other.cols, rows, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, cols/*ldc*/,
                                   1/*commandQueues*/, &queue, 0, nullptr, nullptr);
    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::mul_trans (const ClMatrix& other) const
{
    if (cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {rows, other.rows};

    cl_command_queue queue = clSrvc.queue ();

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasTrans,
                                   rows, other.rows, cols, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, rows/*ldc*/,
                                   1/*commandQueues*/, &queue, 0, nullptr, nullptr);
    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::div (const double scalar) const
{
    return mul (1 / scalar);
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::divisor (const double dividend) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.scalar_div ();

    __(clSetKernelArg (kernel, 0, sizeof (dividend), &dividend));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::el_div (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element division"};

    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.el_div ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::fun (const string& f) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.getFun (f, 1);

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::fun (const string& f, const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element wise function"};

    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.getFun (f, 2);

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));
    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::sigmoid () const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue      = clSrvc.queue ();
    const size_t     workSize[] = {size};

    cl_kernel kernel = clSrvc.sigmoid ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, workSize, nullptr, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________

ClMatrix ClMatrix::sum () const
{
    ClMatrix result {1, 1};

    cl_command_queue queue   = clSrvc.queue ();
    cl_kernel        kernel;

    const size_t vectorSize = clSrvc.vectorSize;
    const size_t compUnits  = clSrvc.compUnits;

    const size_t extra = sizeVec * vectorSize - size;
    if (extra > 0)
    {
        kernel = clSrvc.init_zero ();
        __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
        __(clEnqueueNDRangeKernel (queue, kernel, 1, &size, &extra, nullptr, 0, nullptr, nullptr));
    }

    static ClMatrix tmp {compUnits * vectorSize, 1};

    size_t   globalSize[1];
    size_t   localSize[1];
    cl_mem   src;
    cl_ulong count;
    size_t   localMem;

    size_t wg = clSrvc.sum_comp_unit_Wg;
    if (sizeVec > wg)
    {
        wg = clSrvc.sum_full_load_Wg;
        const cl_uint iter  = ceil ((double) sizeVec / (double) (compUnits * wg));
                      count = sizeVec;

        globalSize[0] = compUnits * wg;  assert (log2 (wg) == ceil (log2 (wg)));
        localSize[0]  = wg;
        localMem      = localSize[0] * vectorSize * sizeof (double);

        kernel = clSrvc.sum_full_load ();

        __(clSetKernelArg (kernel, 0, sizeof (cl_mem),      &mem));
        __(clSetKernelArg (kernel, 1, sizeof (cl_uint),     &iter));
        __(clSetKernelArg (kernel, 2, sizeof (cl_ulong),    &count));
        __(clSetKernelArg (kernel, 3, localMem,             nullptr));
        __(clSetKernelArg (kernel, 4, sizeof (cl_mem),      &tmp.mem));

        __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, globalSize, localSize, 0, nullptr, nullptr));

        globalSize[0] = pow (2, (int) ceil (log2 (compUnits)));  assert (globalSize[0] <= clSrvc.sum_comp_unit_Wg);
        localSize[0]  = globalSize[0];
        src           = tmp.mem;
        count         = compUnits;
        localMem      = tmp.byteSizeVec;
    }
    else
    {
        globalSize[0] = pow (2, (int) ceil (log2 (sizeVec)));  assert (globalSize[0] <= clSrvc.sum_comp_unit_Wg);
        localSize[0]  = globalSize[0];
        src           = mem;
        count         = sizeVec;
        localMem      = localSize[0] * vectorSize * sizeof (double);
    }

    kernel = clSrvc.sum_comp_unit ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem),   &src));
    __(clSetKernelArg (kernel, 1, sizeof (cl_ulong), &count));
    __(clSetKernelArg (kernel, 2, localMem,          nullptr));
    __(clSetKernelArg (kernel, 3, sizeof (cl_mem),   &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, globalSize, localSize, 0, nullptr, nullptr));

    return result;
}
//__________________________________________________________________________________________________
