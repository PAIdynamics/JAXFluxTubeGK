target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/dcomm_handler.cxx
)


target_include_directories(GENEXCore-CXX-Library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

if(GENEX_CXX_ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
