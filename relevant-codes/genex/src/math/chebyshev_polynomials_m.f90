module chebyshev_polynomials_m
    !! Contains functionality to create Chebyshev polynomials

    use genex_fortran_env_m, only: GP
    use genex_status_codes_m, only: GENEX_ERR_MATH

    implicit none
    private

    public :: chebyshev_T, chebyshev_U

contains

    pure function chebyshev_T(n) result(poly)
        !! Returns the first kind Chebyshev T polynomial of order n as a
        !! coefficient array
        integer, intent(in) :: n
        !! Order of the polynomial
        real(kind=GP), dimension(n+1) :: poly
        !! Resulting polynomial coefficients

        poly = chebyshev_base(n, k=1)
    end function

    pure function chebyshev_U(n) result(poly)
        !! Returns the second kind Chebyshev U polynomial of order n as a
        !! coefficient array
        integer, intent(in) :: n
        !! Order of the polynomial
        real(kind=GP), dimension(n+1) :: poly
        !! Resulting polynomial coefficients

        poly = chebyshev_base(n, k=2)
    end function

    recursive pure function chebyshev_base(n, k) result(poly)
        !! Returns the Chebyshev polynomial of kind k and order n as a
        !! coefficient array
        integer, intent(in) :: n
        !! Order of the polynomial
        integer, intent(in) :: k
        !! Kind of the polynomial (1 or 2). If other values are given, all
        !! coefficients will contain the same value set by an error code.
        real(kind=GP), dimension(n+1) :: poly
        !! Resulting polynomial coefficients

        real(kind=GP), dimension(:), allocatable :: c1, c2
        ! Helper polynomials used in the recurrence relation (i.e. Tn and Tn-1)
        integer :: j

        poly = 0.0

        if(k /= 1 .and. k /= 2) then
            ! This case can be easily checked by a calling routine. For these
            ! polynomials it is not possible to have the same coefficients.
            poly = GENEX_ERR_MATH
            return
        endif

        if(n == 0) then
            poly(1) = 1

        else if(n == 1) then
            poly(2) = k

        else
            ! For n > 1 we use the recurrence relation
            allocate(c1(1:(n-1)))
            c1 = chebyshev_base(n - 1, k)
            do j = 1, n
                ! NOTE: Multiplication by x is performed by shifting
                !       the indices in the assignment
                poly(j + 1) = 2.0_GP * c1(j)
            enddo

            allocate(c2(1:(n-2)))
            c2 = chebyshev_base(n - 2, k)
            do j = 1, (n-1)
                poly(j) = poly(j) - c2(j)
            enddo

            deallocate(c1, c2)
        endif

    end function

end module
