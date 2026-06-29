target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_source_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_container_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_lapd_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_dens_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_torque_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_source_heat_cpu_s.f90
)

add_subdirectory(vspectral)

if(ENABLE_UTESTS)
endif()
