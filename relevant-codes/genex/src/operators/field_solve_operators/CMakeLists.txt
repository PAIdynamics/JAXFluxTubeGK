target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_qn_eq_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_qn_eq_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_amps_law_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_amps_law_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_bpar_eq_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_bpar_eq_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_ohms_law_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_solve_ohms_law_s.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
