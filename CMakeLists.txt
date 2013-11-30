cmake_minimum_required(VERSION 2.8.8)

project(cl_matrix)

enable_testing()

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)
find_package(clBLAS REQUIRED)
find_package(Octave REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)

add_custom_command(OUTPUT kernels COMMAND compile_kernels DEPENDS src/kernels.cl)
add_custom_target(do_compile_kernels DEPENDS kernels)
add_dependencies(cl_matrix do_compile_kernels)

file(COPY PKG_ADD DESTINATION ${PROJECT_BINARY_DIR})

add_test(UnitTests ClMatrixTest)
