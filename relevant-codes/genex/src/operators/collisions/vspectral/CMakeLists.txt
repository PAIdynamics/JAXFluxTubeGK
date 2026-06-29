target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_coll_lbd_vspec_cpu_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_coll_vspec_cpu_s.f90
)

if(ENABLE_UTESTS)
endif()
