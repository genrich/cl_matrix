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

    const cl_uint     vectorSize;
    const std::string sfx;
    const cl_uint     compUnits;
    const size_t      globMemKb;
    const size_t      globMemFreeKb;
    const size_t      globMem10Percent;
    const cl::Program program;
    cl::Kernel        init_zero;
    cl::Kernel        uminus;
    cl::Kernel        transpose;
    cl::Kernel        add;
    cl::Kernel        add_scalar;
    cl::Kernel        sub;
    cl::Kernel        scalar_sub;
    cl::Kernel        mul_scalar;
    cl::Kernel        pow_scalar;
    cl::Kernel        scalar_pow;
    cl::Kernel        el_mul;
    cl::Kernel        scalar_div;
    cl::Kernel        el_div;
    cl::Kernel        sigmoid;
    cl::Kernel        sum_full_load;
    cl::Kernel        sum_comp_unit;
    const size_t      sum_full_load_Wg;
    const size_t      sum_comp_unit_Wg;

    const bool        initialized;
    const std::string statusMsg;

                       ClService  ();
                       ~ClService ();
    cl_kernel          getFun     (const std::string&, const int);

    static std::string errMsg     (int);
};

#endif // CL_SERVICE_HPP
