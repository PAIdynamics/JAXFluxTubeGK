module elliptic_integrals_m
    use genex_fortran_env_m, only: DP, GP, DP_EPS
    use, intrinsic :: iso_fortran_env
    use type_converters_m, only: string
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_MATH

    implicit none
    private

    public :: carlson_Rf, carlson_Rd, fukushima_ceik, fukushima_ceie

    real(kind=DP), parameter :: tol = 10.0_DP*DP_EPS
    !! Tolerance
    integer, parameter :: max_iter = 1001
    !! Maximum number of iterations allowed

contains

    function carlson_Rf(arg_a, arg_b, arg_c) result(res)
        !! Computes the Carlson symmetric elliptic integral of the first kind,
        !! Rf up to the precision specified. Algorithm uses iterations and
        !! will terminate with an error if the maximum number of iterations
        !! has been reached.
        !! References:
        !! Carlson (1979) https://doi.org/10.1007/BF01396491
        real(kind=GP), intent(in) :: arg_a, arg_b, arg_c
        !! Arguments of the elliptic integral
        real(kind=GP) :: res
        !! Result

        integer :: n
        !! Loop counter
        real(kind=DP) :: xn, yn, zn
        !! Iterated arguments
        real(kind=DP) :: mu_n
        !! Arithmethic mean of xn, yn, zn
        real(kind=DP) :: X, Y, Z
        !! Fractional differences of xn, yn, zn
        real(kind=DP) :: lambda_n
        !! Sum of pairwise geometric means of fractional differences
        real(kind=DP) :: eps_n
        !! Maximum absolute fractional difference
        real(kind=DP) :: E2, E3
        !! Elementary symmetric functions

        xn = arg_a
        yn = arg_b
        zn = arg_c

        do n = 0, max_iter
            lambda_n = sqrt(xn * yn) + sqrt(xn * zn) + sqrt(yn * zn)
            xn = (xn + lambda_n) / 4.0_DP
            yn = (yn + lambda_n) / 4.0_DP
            zn = (zn + lambda_n) / 4.0_DP

            mu_n = (xn + yn + zn) / 3.0_DP
            X = 1.0_DP - (xn / mu_n)
            Y = 1.0_DP - (yn / mu_n)
            Z = 1.0_DP - (zn / mu_n)
            eps_n = max(abs(X), abs(Y), abs(Z))

            ! Exit if fractional difference becomes smaller than the tolerance.
            ! Then the series expansion is good enough.
            if(eps_n < tol) exit
        enddo

        if (n >= max_iter) then
            call handle_error("Too many iterations in calculation of &
                              &Carlson Rf!", GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        ! NOTE: We use the version of the expansion of Rf in terms of the
        !       elementary symmetric functions E (see 5.10).
        E2 = X * Y - Z**2
        E3 = X * Y * Z
        res = real((  1.0_DP &
                    - 0.1_DP * E2 &
                    + (1.0_DP / 14.0_DP) * E3 &
                    + (1.0_DP / 24.0_DP) * E2**2 &
                    - (3.0_DP / 44.0_DP) * E2 * E3) / sqrt(mu_n), kind=GP)
    end function

    function carlson_Rd(arg_a, arg_b, arg_c) result(res)
        !! Computes the Carlson symmetric elliptic integral of the second kind,
        !! Rd up to the precision specified. Algorithm uses iterations and
        !! will terminate with an error if the maximum number of iterations
        !! has been reached.
        !! References:
        !! Carlson (1979) https://doi.org/10.1007/BF01396491
        real(kind=GP), intent(in) :: arg_a, arg_b, arg_c
        !! Arguments of the elliptic integral
        real(kind=GP) :: res
        !! Result

        integer :: n
        !! Loop counter
        real(kind=DP) :: xn, yn, zn
        !! Iterated arguments
        real(kind=DP) :: mu_n
        !! Arithmethic mean of xn, yn, zn
        real(kind=DP) :: X, Y, Z
        !! Fractional differences of xn, yn, zn
        real(kind=DP) :: lambda_n
        !! Sum of pairwise geometric means of fractional differences
        real(kind=DP) :: eps_n
        !! Maximum absolute fractional difference
        real(kind=DP) :: Bn
        !! Summation term that is added to the result
        real(kind=DP) :: E2, E3, E4, E5
        !! Elementary symmetric functions

        xn = arg_a
        yn = arg_b
        zn = arg_c
        Bn = 0.0_DP

        do n = 0, max_iter
            lambda_n = sqrt(xn * yn) + sqrt(xn * zn) + sqrt(yn * zn)
            Bn = Bn + (4.0_DP**(-n)) / (sqrt(zn) * (zn + lambda_n))
            xn = (xn + lambda_n) / 4.0_DP
            yn = (yn + lambda_n) / 4.0_DP
            zn = (zn + lambda_n) / 4.0_DP

            mu_n = (xn + yn + 3.0_DP * zn) / 5.0_DP
            X = 1.0_DP - (xn / mu_n)
            Y = 1.0_DP - (yn / mu_n)
            Z = 1.0_DP - (zn / mu_n)
            eps_n = max(abs(X), abs(Y), abs(Z))

            ! Exit if fractional difference becomes smaller than the tolerance.
            ! Then the series expansion is good enough.
            if(eps_n < tol) exit
        enddo
        ! NOTE: The counting variable of the loop is needed in the calculation
        !       of a power below. Because we exit before increasing the
        !       counter (in the loop head) we do it manually here, to get the
        !       correct number of iterations executed.
        n = n + 1

        if (n >= max_iter) then
            call handle_error("Too many iterations in calculation of &
                              &Carlson Rd!", GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        ! NOTE: We use the version of the expansion of Rd in terms of the
        !       elementary symmetric functions E (see 5.10).
        E2 = X * Y - 6.0_DP * Z**2
        E3 = (3.0_DP * X * Y - 8.0_DP * Z**2) * Z
        E4 = 3.0_DP * (X * Y - Z**2) * Z**2
        E5 = X * Y * Z**3
        res = real(mu_n**(-1.5_DP) * 4.0_DP**(-n) &
                    * (  1.0_DP &
                      - (3.0_DP / 14.0_DP) * E2 &
                      + (1.0_DP /  6.0_DP) * E3 &
                      + (9.0_DP / 88.0_DP) * E2**2 &
                      - (3.0_DP / 22.0_DP) * E4 &
                      - (9.0_DP / 52.0_DP) * E2 * E3 &
                      + (3.0_DP / 26.0_DP) * E5) + 3.0_DP * Bn, kind=GP)

    end function

    function fukushima_ceik(m_arg) result(res)
        !! Double precision minimax rational approximation of K(m),
        !! the complete elliptic integral of the first kind
        !!
        !! Author: Toshio Fukushima <Toshio.Fukushima@nao.ac.jp>
        !!
        !! Reference: Fukushima (2015) J. Comp. Appl. Math., 282:71--76
        !! "Precise and fast computation of complete elliptic integrals
        !!  by piecewise minimax rational function approximation"
        !! https://doi.org/10.1016/j.cam.2014.12.038
        real(kind=GP) :: m_arg
        !! Argument to evaluate K
        real(kind=GP) :: res
        !! Result

        real(kind=DP) :: mc, ceik, t, t2

        ! NOTE: The complementary argument mc is used in the algorithm
        mc = 1.0_DP - m_arg

        if(mc > 1.0_DP) then
            call handle_error("Fukushima ceik encountered negative values &
                              &for argument m! (was "//string(m_arg, 10)//")", &
                              GENEX_ERR_MATH, __LINE__, __FILE__)
        endif

        if(mc <= 0.0_DP) then
            call handle_error("Fukushima ceik encountered values for argument &
                              &m which are out of range! (was " &
                              //string(m_arg, 10)//")", &
                              GENEX_ERR_MATH, __LINE__, __FILE__)
        endif

#include "fukushima_ceik.txt"

        res = real(ceik, kind=GP)
    end function

    function fukushima_ceie(m_arg) result(res)
        !! Double precision minimax rational approximation of E(m),
        !! the complete elliptic integral of the second kind
        !!
        !! Author: Toshio Fukushima <Toshio.Fukushima@nao.ac.jp>
        !!
        !! Reference: Fukushima (2015) J. Comp. Appl. Math., 282:71--76
        !! "Precise and fast computation of complete elliptic integrals
        !!  by piecewise minimax rational function approximation"
        !! https://doi.org/10.1016/j.cam.2014.12.038
        real(kind=GP) :: m_arg
        !! Argument to evaluate E
        real(kind=GP) :: res
        !! Result

        real(kind=DP) :: mc, ceie, t, t2

        ! NOTE: The complementary argument mc is used in the algorithm
        mc = 1.0_DP - m_arg

        if(mc > 1.0_DP) then
            call handle_error("Fukushima ceie encountered negative values &
                              &for argument m! (was "//string(m_arg, 10)//")", &
                              GENEX_ERR_MATH, __LINE__, __FILE__)
        endif

        if(mc < 0.0_DP) then
            call handle_error("Fukushima ceie encountered values for argument &
                              &m which are out of range! (was " &
                              //string(m_arg, 10)//")", &
                              GENEX_ERR_MATH, __LINE__, __FILE__)
        endif

#include "fukushima_ceie.txt"

        res = real(ceie, kind=GP)
    end function

end module
