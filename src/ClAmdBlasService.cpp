#include <stdio.h>
#include <string.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include <clAmdBlas.h>

#include "ClAmdBlasService.hpp"

using namespace std;

ClAmdBlasService clSrvc;

static cl::Device clDevice()
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    return devices[0];
}

ClAmdBlasService::ClAmdBlasService()
    :device {clDevice()}, ctx {vector<cl::Device>{device}}, queue {ctx, device},
    initialized (clAmdBlasSuccess == clAmdBlasSetup())
{
}

ClAmdBlasService::~ClAmdBlasService()
{
    clAmdBlasTeardown();
}

std::string ClAmdBlasService::errMsg (int code)
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
