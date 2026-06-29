add_pfunit_ctest(operator-utests
    TEST_SOURCES test_rhs_operators.pf
                 test_mom_operators.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

add_pfunit_ctest(boundary-conditions-utests
    TEST_SOURCES  test_bnd_cond_operators.pf
                  test_sheath_operators.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(operator-utests-gpu-cxx
        TEST_SOURCES test_rhs_operators.pf
                     test_mom_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )

    add_pfunit_ctest(boundary-conditions-utests-gpu-cxx
        TEST_SOURCES  test_bnd_cond_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(operator-utests-gpu-acc
        TEST_SOURCES test_rhs_operators.pf
                     test_mom_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )

    add_pfunit_ctest(boundary-conditions-utests-gpu-acc
        TEST_SOURCES  test_bnd_cond_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(operator-utests-gpu-ompx
        TEST_SOURCES test_rhs_operators.pf
                     test_mom_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )

    add_pfunit_ctest(boundary-conditions-utests-gpu-ompx
        TEST_SOURCES  test_bnd_cond_operators.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
