add_pfunit_ctest(devtools-utests
        TEST_SOURCES test_devtools.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(devtools-utests-gpu-cxx
        TEST_SOURCES test_devtools_gpu.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
