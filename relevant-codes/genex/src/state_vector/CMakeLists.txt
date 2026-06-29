target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/state_vector_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/state_vector_s.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
