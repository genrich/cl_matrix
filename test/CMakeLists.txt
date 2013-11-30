include_directories(${OPENCL_INCLUDE_DIRS} ${CLBLAS_INCLUDE_DIRS})

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_executable(ClMatrixTest ClMatrixTest.cpp $<TARGET_OBJECTS:ClMatrix> $<TARGET_OBJECTS:ClService>)
target_link_libraries(ClMatrixTest ${OPENCL_LIBRARIES} ${CLBLAS_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
set_target_properties(ClMatrixTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
