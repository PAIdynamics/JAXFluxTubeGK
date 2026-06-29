target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_0d_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_0d_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_2d_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_2d_cpu_s.f90
)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
