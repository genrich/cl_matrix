#ifndef CL_AMD_BLAS_SERVICE_HPP
#define CL_AMD_BLAS_SERVICE_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class ClAmdBlasService
{

public:
    const cl::Device  device;
    const cl::Context ctx;
    cl::CommandQueue  queue;
    const bool        initialized;

                       ClAmdBlasService  ();
                       ~ClAmdBlasService ();
    static std::string errMsg            (int);
};

#endif // CL_AMD_BLAS_SERVICE_HPP
