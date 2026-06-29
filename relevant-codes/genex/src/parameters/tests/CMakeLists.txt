add_pfunit_ctest(parameter-utests
    TEST_SOURCES test_parameters.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(parameter-utests-gpu-cxx
        TEST_SOURCES test_parameters_cxx.pf
        OTHER_SOURCES res/helpers_parameters_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
    )
endif()
