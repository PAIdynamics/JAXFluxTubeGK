module lagrange_interpolator_m
    !! Module containing prefactors to calculate derivatives on an unstructured
    !! grid based on lagrange interpolation up to fourth order. The grid is
    !! defined by
    !! y1 ---- y2 ---- y3 ---- y4 ---- y5
    use genex_fortran_env_m, only: GP
    implicit none
    private

    public :: monomials_first_derivative
    public :: monomials_second_derivative
    public :: monomials_fourth_derivative

    public :: monomials_first_derivative_backward_2pts
    public :: monomials_first_derivative_forward_2pts

contains

    subroutine monomials_first_derivative(y1, y2, y3, y4, y5, &
                                          m1, m2, m3, m4, m5)
        !! Calculate the monomials of the first derivative of the
        !! interpolating lagrange polynomial evaluated at y3 up to fourth order
        real(kind=GP), intent(in) :: y1, y2, y3, y4, y5
        !! Grid points
        real(kind=GP), intent(out) :: m1, m2, m3, m4, m5
        !! Value of the monomials

        m1 = -(((y2 - y3) * (y3 - y4) * (y3 - y5)) &
             / ((y1 - y2) * (y1 - y3) * (y1 - y4) * (y1 - y5)))
        m2 = ((y1 - y3) * (y3 - y4) * (y3 - y5)) &
             / ((y1 - y2) * (y2 - y3) * (y2 - y4) * (y2 - y5))
        m3 = ((y2 * (3.0_GP * y3**2 + y4 * y5 - 2.0_GP * y3 * (y4 + y5)) &
              + y3 * (-4.0_GP * y3**2 - 2.0_GP * y4 * y5 &
                      + 3.0_GP * y3 * (y4 + y5)) &
              + y1 * (3.0_GP * y3**2 + y4 * y5 - 2.0_GP * y3 * (y4 + y5) &
                      + y2 * (-2.0_GP * y3 + y4 + y5)))) &
             / ((y1 - y3) * (-y2 + y3) * (y3 - y4) * (y3 - y5))
        m4 = ((y1 - y3) * (-y2 + y3) * (y3 - y5)) &
             / ((y1 - y4) * (-y2 + y4) * (-y3 + y4) * (y4 - y5))
        m5 = ((y1 - y3) * (-y2 + y3) * (y3 - y4)) &
             / ((y1 - y5) * (-y2 + y5) * (-y3 + y5) * (-y4 + y5))
    end subroutine

    subroutine monomials_second_derivative(y1, y2, y3, y4, y5, &
                                           m1, m2, m3, m4, m5)
        !! Calculate the monomials of the second derivative of the
        !! interpolating lagrange polynomial evaluated at y3 up to fourth order
        real(kind=GP), intent(in) :: y1, y2, y3, y4, y5
        !! Grid points
        real(kind=GP), intent(out) :: m1, m2, m3, m4, m5
        !! Value of the monomials

        m1 = 2.0_GP * (3.0_GP * y3 * y3 - 2.0_GP * (y2 + y4 + y5) * y3 &
                       + y2 * y4 + (y2 + y4) * y5) &
                    / ((y2 - y1) * (y3 - y1) * (y4 - y1) * (y5 - y1))
        m2 = 2.0_GP * (3.0_GP * y3 * y3 - 2.0_GP * (y1 + y4 + y5) * y3 &
                       + y1 * y4 + (y1 + y4) * y5) &
                    / ((y2 - y1) * (y2 - y3) * (y2 - y4) * (y2 - y5))
        m3 = 2.0_GP * (6.0_GP * y3 * y3 - 3.0_GP * (y1 + y2 + y4 + y5) * y3 &
                       + y1 * y4 + (y1 + y4) * y5 + (y1 + y4 + y5) * y2) &
                    / ((y3 - y1) * (y3 - y2) * (y3 - y4) * (y3 - y5))
        m4 = 2.0_GP * (3.0_GP * y3 * y3 - 2.0_GP * (y1 + y2 + y5) * y3 &
                       + y1 * y2 + (y1 + y2) * y5) &
                    / ((y4 - y1) * (y4 - y2) * (y4 - y3) * (y4 - y5))
        m5 = 2.0_GP * (3.0_GP * y3 * y3 - 2.0_GP * (y1 + y2 + y4) * y3 &
                       + y1 * y2 + (y1 + y2) * y4) &
                    / ((y5 - y1) * (y5 - y2) * (y5 - y3) * (y5 - y4))

    end subroutine

    subroutine monomials_fourth_derivative(y1, y2, y3, y4, y5, &
                                           m1, m2, m3, m4, m5)
        !! Calculate the monomials of the fourth derivative of the
        !! interpolating lagrange polynomial evaluated at y3 up to second order
        real(kind=GP), intent(in) :: y1, y2, y3, y4, y5
        !! Grid points
        real(kind=GP), intent(out) :: m1, m2, m3, m4, m5
        !! Value of the monomials

        m1 =  24.0_GP / ((y1 - y2) * (y1  - y3) * (y1  - y4) * (y1 - y5))
        m2 = -24.0_GP / ((y1 - y2) * (y2  - y3) * (y2  - y4) * (y2 - y5))
        m3 = -24.0_GP / ((y1 - y3) * (-y2 + y3) * (y3  - y4) * (y3 - y5))
        m4 = -24.0_GP / ((y1 - y4) * (-y2 + y4) * (-y3 + y4) * (y4 - y5))
        m5 =  24.0_GP / ((y1 - y5) * (y2  - y5) * (y3  - y5) * (y4 - y5))
    end subroutine

    subroutine monomials_first_derivative_backward_2pts(y1, y2, m1, m2)
        !! Calculate the monomials of the first derivative of the
        !! interpolating lagrange polynomial evaluated at y2 up to first order
        real(kind=GP), intent(in) :: y1, y2
        !! Grid points
        real(kind=GP), intent(out) :: m1, m2
        !! Value of the monomials

        m1 = -1.0_GP / (y2 - y1)
        m2 =  1.0_GP / (y2 - y1)
    end subroutine

    subroutine monomials_first_derivative_forward_2pts(y1, y2, m1, m2)
        !! Calculate the monomials of the first derivative of the
        !! interpolating lagrange polynomial evaluated at y1 up to first order
        real(kind=GP), intent(in) :: y1, y2
        !! Grid points
        real(kind=GP), intent(out) :: m1, m2
        !! Value of the monomials

        ! In this special case, the expression is the same as backward scheme
        ! since m1, m2 flipped + y1, y2 flipped, which lead to the same
        ! expression. And this is a straight line interpolation, so the
        ! derivative is the same everywhere
        m1 = -1.0_GP / (y2 - y1)
        m2 =  1.0_GP / (y2 - y1)
    end subroutine
end module
