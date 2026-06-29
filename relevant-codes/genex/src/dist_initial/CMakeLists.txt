target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_container_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_isotropic_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_anisotropic_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_maxw_cpu_s.f90
	${CMAKE_CURRENT_LIST_DIR}/dist_initial_maxw_vspec_cpu_s.f90
	${CMAKE_CURRENT_LIST_DIR}/dist_initial_maxw_loc_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_bi_maxw_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_double_maxw_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_ring_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/dist_initial_slowing_down_cpu_s.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
