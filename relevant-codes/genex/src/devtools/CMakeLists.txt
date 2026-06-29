target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/devtools_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/device_memory_tracer_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/device_memory_tracer_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/runtime_tracer_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/runtime_tracer_s.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
