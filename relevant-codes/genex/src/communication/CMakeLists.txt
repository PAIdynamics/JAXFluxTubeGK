cmake_policy(SET CMP0076 NEW)
target_sources(GENEXCore-Library
    PRIVATE
    dcomm_handler_m.f90
    dcomm_handler_s.f90
)

if(ENABLE_GPU)
  target_sources(GENEXCore-Library
    PRIVATE
    dcomm_handler_gpu_m.f90
    )
endif()

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
