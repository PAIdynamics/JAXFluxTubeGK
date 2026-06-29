target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bsg_interpolation_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/bsg_interpolation_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/equidistant_interpolation_s.f90
)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
