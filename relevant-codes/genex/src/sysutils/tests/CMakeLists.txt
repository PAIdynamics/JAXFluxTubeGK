add_pfunit_ctest(sysutils-utests
    TEST_SOURCES test_type_converters.pf
    TEST_SOURCES test_system_calls.pf
    TEST_SOURCES test_file_handling.pf
    MAX_PES 1
    LINK_LIBRARIES GENEXCore-Library
)
