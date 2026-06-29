add_pfunit_ctest(field-solver-utests
    TEST_SOURCES test_field_solve_operators.pf
                 test_solve_bpar_eq.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(PACCX_ENABLE_CUDA)
    add_pfunit_ctest(field-solver-utests-gpu-f90-cuda
        TEST_SOURCES test_field_solve_operators.pf
                     test_solve_bpar_eq.pf
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )

    # [ConfiX] Set environment variable to specific test targets
    append_test_environment(TEST field-solver-utests-gpu-f90-cuda
                            ENV "GENEX_UTEST_PARALLAX_GPU_BACKEND=CUDA")

    if(ENABLE_OPENACC)
        add_pfunit_ctest(field-solver-utests-gpu-acc-cuda
            TEST_SOURCES test_field_solve_operators.pf
                         test_solve_bpar_eq.pf
            MAX_PES 8
            LINK_LIBRARIES GENEXCore-Library
            LINK_LIBRARIES GENEXTestUtils-Library
        )

        append_test_environment(TEST field-solver-utests-gpu-acc-cuda
                                ENV "GENEX_UTEST_PARALLAX_GPU_BACKEND=CUDA")

        add_pfunit_ctest(field-solver-utests-gpu-acc-cuda-adv
            TEST_SOURCES test_field_solve_operators.pf
                         test_solve_bpar_eq.pf
            MAX_PES 8
            LINK_LIBRARIES GENEXCore-Library
            LINK_LIBRARIES GENEXTestUtils-Library
        )

        append_test_environment(
            TEST field-solver-utests-gpu-acc-cuda-adv
            ENV "GENEX_UTEST_PARALLAX_GPU_BACKEND=CUDA_ADVANCED")
    endif()

    if(ENABLE_OPENMPX)
        add_pfunit_ctest(field-solver-utests-gpu-ompx-cuda
            TEST_SOURCES test_field_solve_operators.pf
                         test_solve_bpar_eq.pf
            MAX_PES 8
            LINK_LIBRARIES GENEXCore-Library
            LINK_LIBRARIES GENEXTestUtils-Library
        )

        append_test_environment(TEST field-solver-utests-gpu-ompx-cuda
                                ENV "GENEX_UTEST_PARALLAX_GPU_BACKEND=CUDA")

        add_pfunit_ctest(field-solver-utests-gpu-ompx-cuda-adv
            TEST_SOURCES test_field_solve_operators.pf
                         test_solve_bpar_eq.pf
            MAX_PES 8
            LINK_LIBRARIES GENEXCore-Library
            LINK_LIBRARIES GENEXTestUtils-Library
        )

        append_test_environment(
            TEST field-solver-utests-gpu-ompx-cuda-adv
            ENV "GENEX_UTEST_PARALLAX_GPU_BACKEND=CUDA_ADVANCED")
    endif()
endif()
