target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bsg_types_m.f90
)

add_subdirectory(mesh)
add_subdirectory(operators)

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
