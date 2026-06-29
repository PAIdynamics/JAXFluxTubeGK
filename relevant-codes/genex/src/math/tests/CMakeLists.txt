add_pfunit_ctest(math-utests
    TEST_SOURCES test_quadrature.pf
                 test_lagrange_interpolator.pf
                 test_math.pf
                 test_elliptic_integrals.pf
                 test_chebyshev.pf
                 test_polynomial_integrals.pf
                 test_bessel.pf
    MAX_PES 1
    LINK_LIBRARIES GENEXCore-Library
)
