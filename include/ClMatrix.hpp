#ifndef CL_MATRIX_HPP
#define CL_MATRIX_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class ClMatrix
{
    cl_mem mem;

public:
    const int rows;
    const int cols;

public:
              ClMatrix  (const int rows, const int cols);
              ClMatrix  (const int rows, const int cols, const double* data);
              ~ClMatrix ();
              ClMatrix  (ClMatrix const&)                                     = delete;
              ClMatrix  (ClMatrix &&);
    ClMatrix& operator= (const ClMatrix&)                                     = delete;

    void   copyTo   (double*) const;
    size_t byteSize ()        const;

    ClMatrix operator* (const ClMatrix&) const;
    ClMatrix sigmoid   ()                const;
};

#endif // CL_MATRIX_HPP
