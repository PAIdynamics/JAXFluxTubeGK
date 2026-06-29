target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bsg_operators_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/bsg_operators_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/bsg_interpolate_dist_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/bsg_dot_product_s.f90
)

add_subdirectory(interpolation)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
