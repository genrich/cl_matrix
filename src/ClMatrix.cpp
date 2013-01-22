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

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define __(x) do {const int val = (x); if (val) throw runtime_error {"\n" __FILE__ ":" STRINGIZE(__LINE__) ":1 ERROR: " + clSrvc.errMsg (val)};} while (0)

static int fillEvents (cl_event events[], const cl_event event1, const cl_event event2)
{
    int i = 0;
    if (event1 != nullptr) events[i++] = event1;
    if (event2 != nullptr) events[i++] = event2;
    return i;
}

static cl_mem createEmpty (int rows, int cols)
{
    cl_int err;
    cl_mem mem = clCreateBuffer (clSrvc.ctx (), CL_MEM_READ_WRITE, rows * cols * sizeof (double), nullptr, &err);
    if (err != CL_SUCCESS)
        throw runtime_error {"Create memory buffer (" + to_string (err) + ")"};
    return mem;
}

ClMatrix::ClMatrix (const int r, const int c)
    :ClMatrix {r, c, nullptr}
{
}

ClMatrix::ClMatrix (const int r, const int c, const double* data)
    :rows {r}, cols {c}, mem {createEmpty (r, c)}, event {nullptr}
{
    if (data != nullptr)
    {
        const cl_int err = clEnqueueWriteBuffer (clSrvc.queue (), mem, CL_TRUE, 0, r * c * sizeof (double), data, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            throw runtime_error {"Couldn't write to the cl buffer object (" + to_string (err) + ")"};
    }
}

ClMatrix::ClMatrix (ClMatrix&& other)
    :rows {other.rows}, cols {other.cols}, mem {other.mem}, event {other.event}
{
    other.mem   = nullptr;
    other.event = nullptr;
}

ClMatrix::~ClMatrix ()
{
    if (event != nullptr)
    {
        __(clWaitForEvents (1, &event));
        __(clReleaseEvent  (event));
    }
    if (mem != nullptr)
        __(clReleaseMemObject (mem));
}

void ClMatrix::copyTo (double* data) const
{
    int             numEvents = event == nullptr ? 0 : 1;
    const cl_event* events    = numEvents == 0 ? nullptr : &event;

    const cl_int err = clEnqueueReadBuffer (clSrvc.queue (), mem, CL_TRUE, 0, byteSize (), data, numEvents, events, NULL);
    if (err != CL_SUCCESS)
        throw runtime_error {"Couldn't read from the cl buffer object (" + to_string (err) + ")"};
}

size_t ClMatrix::byteSize () const
{
    return rows * cols * sizeof (double);
}

ClMatrix ClMatrix::uminus () const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.uminus ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::transpose () const
{
    ClMatrix result {cols, rows};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.transpose ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (rows),   &rows));
    __(clSetKernelArg (kernel, 2, sizeof (cols),   &cols));
    __(clSetKernelArg (kernel, 3, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::add (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element addition"};

    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    cl_kernel kernel = clSrvc.add ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::add (const double scalar) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.add_scalar ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (scalar), &scalar));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::sub (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element subtraction"};

    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    cl_kernel kernel = clSrvc.sub ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::sub (const double scalar) const
{
    return add (-scalar);
}

ClMatrix ClMatrix::subtrahend (const double minuend) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.scalar_sub ();

    __(clSetKernelArg (kernel, 0, sizeof (minuend), &minuend));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::mul (const ClMatrix& other) const
{
    if (cols != other.rows) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {rows, other.cols};

    cl_command_queue queue     = clSrvc.queue ();
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasNoTrans,
                                   rows, other.cols, cols, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, rows/*ldc*/,
                                   1/*commandQueues*/, &queue, numEvents, events, &result.event);
    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}

ClMatrix ClMatrix::mul (const double scalar) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.mul_scalar ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (scalar), &scalar));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::el_mul (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element multiplication"};

    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    cl_kernel kernel = clSrvc.el_mul ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::trans_mul (const ClMatrix& other) const
{
    if (rows != other.rows) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {cols, other.cols};

    cl_command_queue queue     = clSrvc.queue ();
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasTrans, clAmdBlasNoTrans,
                                   cols, other.cols, rows, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, cols/*ldc*/,
                                   1/*commandQueues*/, &queue, numEvents, events, &result.event);
    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}

ClMatrix ClMatrix::mul_trans (const ClMatrix& other) const
{
    if (cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for multiplication"};

    ClMatrix result {rows, other.rows};

    cl_command_queue queue     = clSrvc.queue ();
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    const auto
        status = clAmdBlasDgemmEx (clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasTrans,
                                   rows, other.rows, cols, 1/*alpha*/,
                                   mem, 0/*offset*/, rows/*lda*/,
                                   other.mem, 0/*offset*/, other.rows/*ldb*/,
                                   0/*beta*/, result.mem, 0/*offset*/, rows/*ldc*/,
                                   1/*commandQueues*/, &queue, numEvents, events, &result.event);
    if (status != clAmdBlasSuccess)
        throw runtime_error {clSrvc.errMsg (status)};

    return result;
}

ClMatrix ClMatrix::div (const double scalar) const
{
    return mul (1 / scalar);
}

ClMatrix ClMatrix::divisor (const double dividend) const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.scalar_div ();

    __(clSetKernelArg (kernel, 0, sizeof (dividend), &dividend));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::el_div (const ClMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) throw runtime_error {"Matrix dimensions mismatch for element division"};

    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    cl_event         evs[2];
    int              numEvents = fillEvents (evs, event, other.event);
    const cl_event*  events    = numEvents == 0 ? nullptr : evs;

    cl_kernel kernel = clSrvc.el_div ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &other.mem));
    __(clSetKernelArg (kernel, 2, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}

ClMatrix ClMatrix::sigmoid () const
{
    ClMatrix result {rows, cols};

    cl_command_queue queue     = clSrvc.queue ();
    size_t           workSize  = rows * cols;
    int              numEvents = event == nullptr ? 0 : 1;
    const cl_event*  events    = numEvents == 0 ? nullptr : &event;

    cl_kernel kernel = clSrvc.sigmoid ();

    __(clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    __(clSetKernelArg (kernel, 1, sizeof (cl_mem), &result.mem));

    __(clEnqueueNDRangeKernel (queue, kernel, 1, nullptr, &workSize, nullptr, numEvents, events, &result.event));

    return result;
}
