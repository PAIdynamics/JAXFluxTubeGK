module polynomial_integrals_m
    !! Contains analytical integrals involving the Hermite and Laguerre
    !! polynomials

    use genex_fortran_env_m, only: GP
    use math_m, only: PI
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_MATH

    implicit none
    private

    public :: integral_hermite, integral_laguerre

contains

    function integral_hermite(p, n, beta) result(res)
        !! Evaluates int_-inf^inf dx H_p(x) x^n exp(-beta * x**2).
        !! Error if p or n are negative integers.
        integer, intent(in) :: p
        !! Order of Hermite polynomial
        integer, intent(in) :: n
        !! Power of x
        real(kind=GP), intent(in) :: beta
        !! Positive scaling factor
        real(kind=GP) :: res
        !! Result of the integral

        integer :: m
        real(kind=GP) :: m_gp, p_gp, n_gp, s_gp, tmp

        p_gp = real(p, kind=GP)
        n_gp = real(n, kind=GP)

        if(p < 0 .or. n < 0) then
            call handle_error("Order p of Hermite polynomial &
                              &and exponent n of x^n must be positive &
                              &integers !", GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        if(beta < 0.0_GP) then
            call handle_error("Scaling factor beta must be positive!", &
                              GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        res = 0.0_GP
        do m = 0, p / 2
            m_gp = real(m, kind=GP)
            s_gp = p_gp - 2.0_GP * m_gp + n_gp
            tmp = 0.0_GP
            if(mod(s_gp, 2.0_GP) == 0.0_GP) then
                tmp = Gamma(s_gp + 1.0_GP) * sqrt(PI) &
                    / 2.0_GP**(s_gp + 1) / Gamma(s_gp / 2.0_GP + 1.0_GP)
            else
                tmp = 0.0_GP
            endif

            res  = res + (-1.0_GP)**m / Gamma(m_gp + 1.0_GP) &
                 / Gamma(p_gp - 2.0_GP * m_gp + 1.0_GP) &
                 * sqrt(1.0 / beta)**(p - 2 * m + n + 1) &
                 * 2.0_GP**(p - 2 * m) * tmp
        enddo
        res = res * 2.0_GP * PI &
            * sqrt(Gamma(p_gp + 1.0_GP) / 2.0_GP**p)

    end function

    function integral_laguerre(j, n, beta) result(res)
        !! Evaluates int_0^inf dx L_j(x) x^n exp(-beta * x).
        !! Error if p or n are negative integers.
        integer, intent(in) :: j
        !! Order of Laguerre polynomial
        integer, intent(in) :: n
        !! Power of x
        real(kind=GP), intent(in) :: beta
        !! Scaling factor
        real(kind=GP) :: res
        !! Result of the integral

        integer :: k
        real(kind=GP) :: k_gp, n_gp, j_gp

        if(j < 0 .or. n < 0) then
            call handle_error("Order j of Laguerre polynomial &
                              &and exponent of x^n must be positive &
                              &integers !", GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        if(beta < 0.0_GP) then
            call handle_error("Scaling factor beta must be positive!", &
                              GENEX_ERR_MATH, &
                              __LINE__, __FILE__)
        endif

        n_gp = real(n, kind=GP)
        j_gp = real(j, kind=GP)

        res = 0.0_GP
        do k = 0, int(j)
            k_gp = real(k, kind=GP)
            res = res &
                + (-1.0_GP)**k &
                * (1.0_GP / beta)**(k + n + 1) &
                * Gamma(k_gp + n_gp + 1.0_GP) &
                / Gamma(k_gp + 1.0_GP) / Gamma(k_gp + 1.0_GP) &
                / Gamma(j_gp - k_gp + 1.0_GP)
        enddo
        res = res * Gamma(j_gp + 1.0_GP)
    end function

end module
