add_pfunit_ctest(dcomm-handler-utests
    TEST_SOURCES test_dcomm_handler.pf
    OTHER_SOURCES res/helpers_dcomm_handler.f90
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
)

if(ENABLE_GPU)
    target_sources(dcomm-handler-utests
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_dcomm_handler_gpu_m.f90
    )

    add_pfunit_ctest(dcomm-handler-utests-gpu-cxx
        TEST_SOURCES test_dcomm_handler_gpu.pf
                     test_dcomm_handler.pf
        OTHER_SOURCES res/helpers_dcomm_handler_gpu_m.f90
                      res/helpers_dcomm_handler.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(dcomm-handler-utests-gpu-acc
        TEST_SOURCES test_dcomm_handler_gpu.pf
                     test_dcomm_handler.pf
        OTHER_SOURCES res/helpers_dcomm_handler_gpu_m.f90
                      res/helpers_dcomm_handler.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(dcomm-handler-utests-gpu-ompx
        TEST_SOURCES test_dcomm_handler_gpu.pf
                     test_dcomm_handler.pf
        OTHER_SOURCES res/helpers_dcomm_handler_gpu_m.f90
                      res/helpers_dcomm_handler.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
