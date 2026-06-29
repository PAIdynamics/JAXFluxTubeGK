add_pfunit_ctest(bsg-types-utests
    TEST_SOURCES test_bsg_types.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
