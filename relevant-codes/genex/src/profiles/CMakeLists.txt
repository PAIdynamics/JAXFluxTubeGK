target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/profiles_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_lapd_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_cbc_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_sine_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_uniform_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_plateau_sine_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profile_container_m.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
