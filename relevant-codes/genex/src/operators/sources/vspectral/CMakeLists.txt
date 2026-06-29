target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_source_dens_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_heat_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_torque_vspec_cpu_s.f90
)

if(ENABLE_UTESTS)
endif()
