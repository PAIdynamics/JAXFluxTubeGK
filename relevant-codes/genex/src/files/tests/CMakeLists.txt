add_pfunit_ctest(files-utests
    TEST_SOURCES test_part_handler.pf
                 test_diag_files.pf
    OTHER_SOURCES res/helpers_files.f90
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)
