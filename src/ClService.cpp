#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

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
        message = string (e.what ()) + ", " + ClService::errMsg (errCode);
    }
    catch (exception &e)
    {
        errCode = 1;
        message = e.what ();
    }
    return {};
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

ClService::ClService ()
    :errCode {0}, message {""},
    device {clDevice ()}, devices {vector<cl::Device> {device}}, ctx {devices}, queue {ctx, device},
    program {loadProgram ("kernels", errCode, message)},
    el_mul  {loadKernel (program, "el_mul",  errCode, message)},
    sigmoid {loadKernel (program, "sigmoid", errCode, message)},
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
        case clAmdBlasInvalidValue:         return "CL_INVALID_VALUE";
        case clAmdBlasInvalidCommandQueue:  return "CL_INVALID_COMMAND_QUEUE";
        case clAmdBlasInvalidContext:       return "CL_INVALID_CONTEXT";
        case clAmdBlasInvalidMemObject:     return "CL_INVALID_MEM_OBJECT";
        case clAmdBlasInvalidDevice:        return "CL_INVALID_DEVICE";
        case clAmdBlasInvalidEventWaitList: return "CL_INVALID_EVENT_WAIT_LIST";
        case clAmdBlasOutOfResources:       return "CL_OUT_OF_RESOURCES";
        case clAmdBlasOutOfHostMemory:      return "CL_OUT_OF_HOST_MEMORY";
        case clAmdBlasInvalidOperation:     return "CL_INVALID_OPERATION";
        case clAmdBlasCompilerNotAvailable: return "CL_COMPILER_NOT_AVAILABLE";
        case clAmdBlasBuildProgramFailure:  return "CL_BUILD_PROGRAM_FAILURE";
        case clAmdBlasNotImplemented:       return "Functionality is not implemented";
        case clAmdBlasNotInitialized:       return "clAmdBlas library is not initialized yet";
        case clAmdBlasInvalidMatA:          return "Matrix A is not a valid memory object";
        case clAmdBlasInvalidMatB:          return "Matrix B is not a valid memory object";
        case clAmdBlasInvalidMatC:          return "Matrix C is not a valid memory object";
        case clAmdBlasInvalidVecX:          return "Vector X is not a valid memory object";
        case clAmdBlasInvalidVecY:          return "Vector Y is not a valid memory object";
        case clAmdBlasInvalidDim:           return "An input dimension (M,N,K) is invalid";
        case clAmdBlasInvalidLeadDimA:      return "Leading dimension A must not be less than the size of the first dimension";
        case clAmdBlasInvalidLeadDimB:      return "Leading dimension B must not be less than the size of the second dimension";
        case clAmdBlasInvalidLeadDimC:      return "Leading dimension C must not be less than the size of the third dimension";
        case clAmdBlasInvalidIncX:          return "The increment for a vector X must not be 0";
        case clAmdBlasInvalidIncY:          return "The increment for a vector Y must not be 0";
        case clAmdBlasInsufficientMemMatA:  return "The memory object for Matrix A is too small";
        case clAmdBlasInsufficientMemMatB:  return "The memory object for Matrix B is too small";
        case clAmdBlasInsufficientMemMatC:  return "The memory object for Matrix C is too small";
        case clAmdBlasInsufficientMemVecX:  return "The memory object for Vector X is too small";
        case clAmdBlasInsufficientMemVecY:  return "The memory object for Vector Y is too small";
        default:                            return "Error message not found, code=" + std::to_string (code);
    }
}
