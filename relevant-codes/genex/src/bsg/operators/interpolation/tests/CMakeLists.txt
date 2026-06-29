add_pfunit_ctest(bsg-interpolation-utests
    TEST_SOURCES test_bsg_interpolation.pf
    MAX_PES 1
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
