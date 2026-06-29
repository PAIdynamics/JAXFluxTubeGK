target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/genex_error_handling_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/genex_status_codes_m.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
