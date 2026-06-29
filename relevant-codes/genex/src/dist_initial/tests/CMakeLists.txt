add_pfunit_ctest(dist_initial-utests
    TEST_SOURCES test_dist_initial.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
