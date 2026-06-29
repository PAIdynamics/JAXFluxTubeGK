target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/block_structured_grids_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/block_structured_grids_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/bsg_construct_s.f90
)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
