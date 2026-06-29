add_pfunit_ctest(diagnostics-utests
    TEST_SOURCES test_diagnostic_operators.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
