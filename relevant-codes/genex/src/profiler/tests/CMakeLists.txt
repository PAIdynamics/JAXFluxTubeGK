add_pfunit_ctest(profiler-utests
    TEST_SOURCES test_profiler.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(profiler-utests-gpu-cxx
        TEST_SOURCES test_profiler_gpu.pf
                     test_profiler.pf
        OTHER_SOURCES res/helpers_profiler_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
