target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_0d_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_diag_mom_2d_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_maxwells_eq_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_ohms_law_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_rhs_vlasov_eq_dynamic_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_set_initial_condition_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_rhs_vlasov_eq_static_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_rhs_vlasov_eq_static_vspec_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_bnd_cond_neum_vspec_cpu_s.f90
)


if(ENABLE_UTESTS)
endif()
