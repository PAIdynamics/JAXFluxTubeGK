target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_magfield_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_parcon_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_vspec_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/structured_grids_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/mu_grid_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/vp_grid_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/phi_grid_s.f90
)

if(ENABLE_GPU)
    target_sources(GENEXCore-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_gpu_m.f90
            ${CMAKE_CURRENT_LIST_DIR}/mesh_5d_gpu_s.f90
    )
endif()

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
