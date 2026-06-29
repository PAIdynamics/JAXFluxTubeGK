target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_omp.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_copy.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_copy.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_copy_omp.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_axpy.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_axpy.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_axpy_omp.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb.hxx
        ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_omp.hxx
)

if(ENABLE_OPENACC)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_acc.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_copy_acc.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_axpy_acc.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_acc.hxx
    )
endif()

if(ENABLE_OPENMPX)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_ompx.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_copy_ompx.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_axpy_ompx.hxx
            ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_ompx.hxx
    )
endif()

# NOTE: The following CUDA-support is only experimental feature to ensure
#       that GENE-X build configuration is CUDA-compatible. This feature is
#       not meant for production run unless stated otherwise in the future.
if(ENABLE_CUDA)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_cuda.cuh
            ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_cuda.cu
            ${CMAKE_CURRENT_LIST_DIR}/op_copy_cuda.cuh
            ${CMAKE_CURRENT_LIST_DIR}/op_copy_cuda.cu
            ${CMAKE_CURRENT_LIST_DIR}/op_axpy_cuda.cuh
            ${CMAKE_CURRENT_LIST_DIR}/op_axpy_cuda.cu
            ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_cuda.cuh
            ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_cuda.cu
    )
endif()

target_include_directories(GENEXCore-CXX-Library
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
