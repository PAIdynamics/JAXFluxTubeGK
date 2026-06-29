target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/logger_m.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
