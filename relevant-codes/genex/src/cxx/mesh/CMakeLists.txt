target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/csrmat_genex.cxx
        ${CMAKE_CURRENT_LIST_DIR}/mesh_5d.cxx
)

target_include_directories(GENEXCore-CXX-Library
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
