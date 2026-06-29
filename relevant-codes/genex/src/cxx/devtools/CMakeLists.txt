target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/device_memory_tracer.cxx
        ${CMAKE_CURRENT_LIST_DIR}/runtime_tracer.cxx
)

target_include_directories(GENEXCore-CXX-Library
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
