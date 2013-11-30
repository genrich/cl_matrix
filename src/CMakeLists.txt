include_directories(${OPENCL_INCLUDE_DIRS} ${CLBLAS_INCLUDE_DIRS} ${OCTAVE_INCLUDE_DIRS})

add_library(ClMatrix  OBJECT ClMatrix.cpp)
add_library(ClService OBJECT ClService.cpp)

add_executable(compile_kernels compile_kernels.cpp $<TARGET_OBJECTS:ClService>)
target_link_libraries(compile_kernels ${OPENCL_LIBRARIES} ${CLBLAS_LIBRARIES})
file(COPY kernels.cl DESTINATION .)

add_library(cl_matrix SHARED octave_cl_matrix.cpp ClMatrix.cpp ClService.cpp)
target_link_libraries(cl_matrix ${OCTAVE_LIBRARIES} ${OPENCL_LIBRARIES} ${CLBLAS_LIBRARIES})
set_target_properties(cl_matrix PROPERTIES PREFIX ""
                                           SUFFIX ".oct"
                                           LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
