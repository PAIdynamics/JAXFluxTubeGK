add_pfunit_ctest(bsg-operators-utests
    TEST_SOURCES test_bsg_operators.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
