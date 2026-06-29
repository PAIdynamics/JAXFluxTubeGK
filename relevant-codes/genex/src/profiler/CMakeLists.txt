target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/profiler_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/profiler_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/profiler_region_s.f90
)

if(ENABLE_GPU)
  target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/profiler_gpu_m.f90
  )
endif()

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
