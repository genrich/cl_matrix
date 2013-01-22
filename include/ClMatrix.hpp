#ifndef CL_MATRIX_HPP
#define CL_MATRIX_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class ClMatrix
{
public:
    const int rows;
    const int cols;

private:
    cl_mem   mem;
    cl_event event;

public:
    ClMatrix uminus     ()                const;
    ClMatrix transpose  ()                const;
    ClMatrix add        (const ClMatrix&) const;
    ClMatrix add        (const double)    const;
    ClMatrix sub        (const ClMatrix&) const;
    ClMatrix sub        (const double)    const;
    ClMatrix subtrahend (const double)    const;
    ClMatrix mul        (const ClMatrix&) const;
    ClMatrix mul        (const double)    const;
    ClMatrix el_mul     (const ClMatrix&) const;
    ClMatrix trans_mul  (const ClMatrix&) const;
    ClMatrix mul_trans  (const ClMatrix&) const;
    ClMatrix div        (const double)    const;
    ClMatrix divisor    (const double)    const;
    ClMatrix el_div     (const ClMatrix&) const;
public:
              ClMatrix  (const int rows, const int cols);
              ClMatrix  (const int rows, const int cols, const double* data);
              ~ClMatrix ();
              ClMatrix  (ClMatrix const&)                                     = delete;
              ClMatrix  (ClMatrix &&);
    ClMatrix& operator= (const ClMatrix&)                                     = delete;

    void   copyTo   (double*) const;
    size_t byteSize ()        const;

    ClMatrix sigmoid   ()                const;
};

#endif // CL_MATRIX_HPP
