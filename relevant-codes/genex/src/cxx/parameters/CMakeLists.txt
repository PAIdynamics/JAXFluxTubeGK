target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/params_gpu_offload.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_collisions.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_devtools.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_gyrokinetic_system.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_normalization.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_numerical_scheme.cxx
        ${CMAKE_CURRENT_LIST_DIR}/params_species.cxx
)

if(ENABLE_OPENACC)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/params_gpu_offload_acc.cxx
    )
endif()

if(ENABLE_OPENMPX)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/params_gpu_offload_ompx.cxx
    )
endif()

if(ENABLE_CUDA)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/params_gpu_offload_cuda.cu
    )
endif()

target_include_directories(GENEXCore-CXX-Library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
