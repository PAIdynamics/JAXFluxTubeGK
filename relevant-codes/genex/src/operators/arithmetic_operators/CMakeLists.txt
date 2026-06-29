target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_copy_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_copy_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_axpy_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_axpy_cpu_s.f90
)

if(ENABLE_GPU)
    target_sources(GENEXCore-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/op_set_uniform_gpu_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/op_copy_gpu_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/op_axpy_gpu_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/op_lin_comb_gpu_s.f90
    )
endif()

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
