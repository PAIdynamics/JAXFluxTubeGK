target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/params_source_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/params_source_lapd_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/params_source_dens_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/params_source_torque_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/params_source_heat_m.f90
)
