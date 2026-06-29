target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_bgk_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_lbd_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_lorentz_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_fpl_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_coll_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_coll_cpu_s.f90
)

if(ENABLE_GPU)
    target_sources(GENEXCore-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/op_mom_coll_gpu_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/op_coll_bgk_gpu_s.f90
    )
endif()

add_subdirectory(vspectral)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
