add_pfunit_ctest(profiles-utests
    TEST_SOURCES test_profiles.pf
    TEST_SOURCES test_profiles_neutrals.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
