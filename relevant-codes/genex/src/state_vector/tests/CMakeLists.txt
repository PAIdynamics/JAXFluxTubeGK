add_pfunit_ctest(state-vector-utests
    TEST_SOURCES test_state_vector.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(state-vector-utests-gpu-cxx
        TEST_SOURCES test_state_vector.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(state-vector-utests-gpu-acc
        TEST_SOURCES test_state_vector.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(state-vector-utests-gpu-ompx
        TEST_SOURCES test_state_vector.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
