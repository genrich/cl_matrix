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
    cl::Kernel        sigmoid;

    const bool        initialized;
    const std::string statusMsg;

                       ClService  ();
                       ~ClService ();
    static std::string errMsg     (int);
};

#endif // CL_SERVICE_HPP
