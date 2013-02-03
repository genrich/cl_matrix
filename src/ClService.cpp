#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>

#include "ClService.hpp"

#include <clAmdBlas.h>

using namespace std;

ClService clSrvc;

static cl::Device clDevice ()
{
    vector<cl::Platform> platforms;
    cl::Platform::get (&platforms);

    vector<cl::Device> devices;
    platforms [0].getDevices (CL_DEVICE_TYPE_GPU, &devices);
    return devices [0];
}

static cl::Program loadProgram (string programName, int& errCode, string& message)
{
    if (0 != errCode)
        return {};

    try
    {
        ifstream kernelsFile {programName, ios::binary};
        if (!kernelsFile.is_open ()) throw runtime_error {"No compiled 'kernels' file found."};
        vector<char> buffer {istreambuf_iterator<char> {kernelsFile}, istreambuf_iterator<char> {}};
        cl::Program::Binaries binaries {make_pair (buffer.data (), buffer.size ())};
        cl::Program program {clSrvc.ctx, clSrvc.devices, binaries};
        program.build ();
        return program;
    }
    catch (exception& e)
    {
        errCode = 1;
        message = e.what ();
    }
    return {};
}

static cl::Kernel loadKernel (const cl::Program& program, string kernelName, int& errCode, string& message)
{
    if (0 != errCode)
        return {};

    try
    {
        return {program, kernelName.c_str ()};
    }
    catch (cl::Error &e)
    {
        errCode = 1;
        message = "Load kernel \"" + kernelName + "\" error, " + string (e.what ()) + ", " + ClService::errMsg (errCode);
    }
    catch (exception &e)
    {
        errCode = 1;
        message = "Load kernel \"" + kernelName + "\" error, " + e.what ();
    }
    return {};
}

template <cl_int name>
static size_t wgInfo (const cl::Kernel& kernel, const cl::Device& device, int& errCode)
{
    if (0 != errCode)
        return {};

    return kernel.getWorkGroupInfo<name> (device);
}

static void check (const bool condition, const string msg, int& errCode, string& message)
{
    if (0 != errCode)
        return;

    if (!condition)
    {
        errCode = 1;
        message = msg;
    }
}

bool initialize (int& errCode, string& message)
{
    if (0 != errCode)
        return false;

    errCode = clAmdBlasSetup ();
    if (clAmdBlasSuccess == errCode)
    {
        message = "ok";
        return true;
    }
    else
    {
        message = ClService::errMsg (errCode);
        return false;
    }
}

ClService::ClService ():
    errCode {0},
    message {""},
    device  {clDevice ()},
    devices {vector<cl::Device> {device}},
    ctx     {devices},
    queue   {ctx, device},

    vectorSize       {device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE> () == 2U ? 2U : 1U}, // fallback to 1 if 2 is not supported
    sfx              {vectorSize > 1 ? "_" + to_string (vectorSize) : ""},
    compUnits        {device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS> ()},
    globMemKb        {device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE> () / 1024},
    globMemFreeKb    {device.getInfo<CL_DEVICE_GLOBAL_FREE_MEMORY_AMD> ()[0]},
    globMem10Percent {(check (globMemFreeKb > globMemKb * 9.0 / 10.0 && globMemFreeKb < globMemKb,
                              "Invalid free mem(" + to_string (globMemFreeKb) + " Kb) size on the device with glob mem(" + to_string (globMemKb) + " Kb)",
                              errCode, message),
                      static_cast<size_t> (globMemKb / 10.0))},

    program {loadProgram ("kernels", errCode, message)},

    init_zero   {loadKernel (program, "init_zero",  errCode, message)},
    uminus      {loadKernel (program, "uminus",     errCode, message)},
    transpose   {loadKernel (program, "transpose",  errCode, message)},
    add         {loadKernel (program, "add",        errCode, message)},
    add_scalar  {loadKernel (program, "add_scalar", errCode, message)},
    sub         {loadKernel (program, "sub",        errCode, message)},
    scalar_sub  {loadKernel (program, "scalar_sub", errCode, message)},
    mul_scalar  {loadKernel (program, "mul_scalar", errCode, message)},
    el_mul      {loadKernel (program, "el_mul",     errCode, message)},
    scalar_div  {loadKernel (program, "scalar_div", errCode, message)},
    el_div      {loadKernel (program, "el_div",     errCode, message)},
    sigmoid     {loadKernel (program, "sigmoid",    errCode, message)},

    sum_full_load    {loadKernel (program, "sum_full_load" + sfx, errCode, message)},
    sum_comp_unit    {loadKernel (program, "sum_comp_unit" + sfx, errCode, message)},
    sum_full_load_Wg {wgInfo<CL_KERNEL_WORK_GROUP_SIZE> (sum_full_load, device, errCode)},
    sum_comp_unit_Wg {wgInfo<CL_KERNEL_WORK_GROUP_SIZE> (sum_comp_unit, device, errCode)},

    initialized {initialize (errCode, message)},
    statusMsg   {message}

{
}

ClService::~ClService ()
{
    clAmdBlasTeardown ();
}

std::string ClService::errMsg (int code)
{
    switch (code)
    {
        case CL_SUCCESS:                                   return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                          return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:                      return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:                    return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:             return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                          return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                        return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:              return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                          return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:                     return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:                     return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                               return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:              return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:                   return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:                      return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:                      return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:                   return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:             return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:                             return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                       return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                          return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                           return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:                  return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:                     return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                          return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                        return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:           return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                        return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                           return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:                     return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                           return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:                return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                       return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:                 return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                         return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                         return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                          return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                       return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:                    return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:                   return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:                    return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:                     return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:                   return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                             return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                         return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                         return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                       return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                         return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:                  return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:                          return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:                  return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:                  return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:                    return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        // case clAmdBlasInvalidValue:                        return "CL_INVALID_VALUE";
        // case clAmdBlasInvalidCommandQueue:                 return "CL_INVALID_COMMAND_QUEUE";
        // case clAmdBlasInvalidContext:                      return "CL_INVALID_CONTEXT";
        // case clAmdBlasInvalidMemObject:                    return "CL_INVALID_MEM_OBJECT";
        // case clAmdBlasInvalidDevice:                       return "CL_INVALID_DEVICE";
        // case clAmdBlasInvalidEventWaitList:                return "CL_INVALID_EVENT_WAIT_LIST";
        // case clAmdBlasOutOfResources:                      return "CL_OUT_OF_RESOURCES";
        // case clAmdBlasOutOfHostMemory:                     return "CL_OUT_OF_HOST_MEMORY";
        // case clAmdBlasInvalidOperation:                    return "CL_INVALID_OPERATION";
        // case clAmdBlasCompilerNotAvailable:                return "CL_COMPILER_NOT_AVAILABLE";
        // case clAmdBlasBuildProgramFailure:                 return "CL_BUILD_PROGRAM_FAILURE";
        case clAmdBlasNotImplemented:                      return "Functionality is not implemented";
        case clAmdBlasNotInitialized:                      return "clAmdBlas library is not initialized yet";
        case clAmdBlasInvalidMatA:                         return "Matrix A is not a valid memory object";
        case clAmdBlasInvalidMatB:                         return "Matrix B is not a valid memory object";
        case clAmdBlasInvalidMatC:                         return "Matrix C is not a valid memory object";
        case clAmdBlasInvalidVecX:                         return "Vector X is not a valid memory object";
        case clAmdBlasInvalidVecY:                         return "Vector Y is not a valid memory object";
        case clAmdBlasInvalidDim:                          return "An input dimension (M,N,K) is invalid";
        case clAmdBlasInvalidLeadDimA:                     return "Leading dimension A must not be less than the size of the first dimension";
        case clAmdBlasInvalidLeadDimB:                     return "Leading dimension B must not be less than the size of the second dimension";
        case clAmdBlasInvalidLeadDimC:                     return "Leading dimension C must not be less than the size of the third dimension";
        case clAmdBlasInvalidIncX:                         return "The increment for a vector X must not be 0";
        case clAmdBlasInvalidIncY:                         return "The increment for a vector Y must not be 0";
        case clAmdBlasInsufficientMemMatA:                 return "The memory object for Matrix A is too small";
        case clAmdBlasInsufficientMemMatB:                 return "The memory object for Matrix B is too small";
        case clAmdBlasInsufficientMemMatC:                 return "The memory object for Matrix C is too small";
        case clAmdBlasInsufficientMemVecX:                 return "The memory object for Vector X is too small";
        case clAmdBlasInsufficientMemVecY:                 return "The memory object for Vector Y is too small";
        default:                                           return "Error message not found, code=" + std::to_string (code);
    }
}
