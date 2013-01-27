#ifndef CL_SERVICE_HPP
#define CL_SERVICE_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class ClService
{
    int         errCode;
    std::string message;

public:
    const cl::Device              device;
    const std::vector<cl::Device> devices;
    const cl::Context             ctx;
    cl::CommandQueue              queue;

    const cl::Program program;
    cl::Kernel        uminus;
    cl::Kernel        transpose;
    cl::Kernel        add;
    cl::Kernel        add_scalar;
    cl::Kernel        sub;
    cl::Kernel        scalar_sub;
    cl::Kernel        mul_scalar;
    cl::Kernel        el_mul;
    cl::Kernel        scalar_div;
    cl::Kernel        el_div;
    cl::Kernel        sigmoid;

    const bool        initialized;
    const std::string statusMsg;

                       ClService  ();
                       ~ClService ();
    static std::string errMsg     (int);

    const size_t globMem10Percent;
};

#endif // CL_SERVICE_HPP
