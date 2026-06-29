target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/type_converters_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/system_calls_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/file_handling_m.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
