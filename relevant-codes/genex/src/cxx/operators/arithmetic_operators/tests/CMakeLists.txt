target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/helpers_arithmetic_operators.cxx
)

if(ENABLE_CUDA)
    target_sources(GENEXCore-CXX-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/helpers_arithmetic_operators.cu
    )
endif()
