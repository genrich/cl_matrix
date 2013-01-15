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
    bool      transpose;

public:
              ClMatrix  (const int rows, const int cols);
              ClMatrix  (const int rows, const int cols, const double* data);
              ~ClMatrix ();
              ClMatrix  (ClMatrix const&)                                     = delete;
              ClMatrix  (ClMatrix &&);
    ClMatrix& operator= (const ClMatrix&)                                     = delete;
    ClMatrix  operator* (const ClMatrix&)                                     const;
    void      copyTo    (double*)                                             const;
};

#endif // CL_MATRIX_HPP
