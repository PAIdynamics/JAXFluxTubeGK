module quadrature_m
    use genex_fortran_env_m, only: GP
    ! From PARALLAX
    use gauss_quadrature_m, only: gauss_laguerre

    implicit none
    private

    public :: create_midpoint_weights, create_trapezoidal_weights, &
              create_simpson_weights, create_gauss_laguerre

contains

    subroutine create_midpoint_weights(npoints, points, weights, quadratic)
        !! Initializes weights for the midpoint integration method. Is
        !! intended to be used for uniform one/two sided grids. Can be used
        !! with one-sided quadratic grids if the optional quadratic is set.
        !!
        !! NOTE: Does not require a staggered grid for the calculation of
        !!       the weights, although the use of a staggered or not changes
        !!       the integral that is calculated (different boundaries).
        integer, intent(in) :: npoints
        !! Number of elements in the grid
        real(kind=GP), dimension(1:npoints), intent(in) :: points
        !! Values of the gridpoints
        real(kind=GP), dimension(1:npoints), intent(out) :: weights
        !! Weights for numerical integration
        logical, optional, intent(in) :: quadratic
        !! If true, midpoint weights for one-sided quadratic grid are returned.
        !! In that case the lower bound of the integral is exactly zero.
        !! (default = false)

        logical :: quadratic_local
        !! Local variables for optionals
        integer :: i
        !! Loop index

        if(present(quadratic)) then
            quadratic_local = quadratic
        else
            quadratic_local = .false.
        endif

        if(npoints == 1) then
            ! In the case of one point, no integration is performed.
            weights(1) = 1.0_GP
        else
            if(quadratic_local) then
                ! NOTE: This version only works for one-sided grids. This
                !       implies that the lower bound of the integral is equal
                !       to zero.

                ! Equivalent to performing change of variables to sqrt(x) and
                ! performing integration using the midpoint rule.
                do i = 1, npoints
                    weights(i) = 2.0_GP * sqrt(points(i)) &
                               * (sqrt(points(2)) - sqrt(points(1)))
                end do
            else
                weights = points(2) - points(1)
            endif
        endif
    end subroutine

    subroutine create_trapezoidal_weights(npoints, points, weights)
        !! Initializes weights for the trapezoidal integration method. Works
        !! with both, uniform and non-uniform grids.
        integer, intent(in) :: npoints
        !! Number of elements in the grid
        real(kind=GP), dimension(1:npoints), intent(in) :: points
        !! Values of the gridpoints
        real(kind=GP), dimension(1:npoints), intent(out) :: weights
        !! Weights for numerical integration

        integer :: i
        !! Loop index

        if(npoints == 1) then
            ! In the case of one point, no integration is performed
            weights(1) = 1.0_GP
        else
            ! Edge points
            weights(1)       = 0.5_GP * (points(2) - points(1))
            weights(npoints) = 0.5_GP * (points(npoints) - points(npoints - 1))
            ! Inner points
            do i = 2, (npoints - 1)
                weights(i) = 0.5_GP * (points(i + 1) - points(i - 1))
            end do
        endif
    end subroutine

    subroutine create_simpson_weights(npoints, points, weights, ierr)
        !! Initializes weights for the Simpson integration method for a
        !! uniform grid. Does not check for equal spacing between points.
        !! Requires at least 8 grid points.
        integer, intent(in) :: npoints
        !! Number of elements in the grid
        real(kind=GP), dimension(1:npoints), intent(in) :: points
        !! Values of the gridpoints
        real(kind=GP), dimension(1:npoints), intent(out) :: weights
        !! Weights for numerical integration
        integer, intent(out) :: ierr
        !! Error code (0 if weights created, 1 otherwise)

        real(kind=GP) :: delta
        ! Spacing between grid points

        if(npoints < 8) then
            ! We need at least 8 points for this method
            ierr = 1
        else
            ! Calculate spacing from first 2 points (assume uniform grid)
            delta = points(2) - points(1)
            ! The bulk of the domain has weights 1
            weights = 1.0_GP
            ! Treat the edge of the integration domain with different weights
            weights(1)         = 17.0_GP / 48.0_GP
            weights(2)         = 59.0_GP / 48.0_GP
            weights(3)         = 43.0_GP / 48.0_GP
            weights(4)         = 49.0_GP / 48.0_GP
            weights(npoints-3) = 49.0_GP / 48.0_GP
            weights(npoints-2) = 43.0_GP / 48.0_GP
            weights(npoints-1) = 59.0_GP / 48.0_GP
            weights(npoints)   = 17.0_GP / 48.0_GP
            ! Apply spacing to weights
            weights = weights * delta
            ierr = 0
        endif

    end subroutine

    subroutine create_gauss_laguerre(npoints, length, points, weights, ierr)
        !! Initializes points and weights for the Gauss-Laguerre quadrature.
        integer, intent(in) :: npoints
        !! Number of elements in the grid
        real(kind=GP), intent(in) :: length
        !! Length of the grid
        real(kind=GP), dimension(1:npoints), intent(out) :: points
        !! Values of the gridpoints
        real(kind=GP), dimension(1:npoints), intent(out) :: weights
        !! Weights for numerical integration
        integer, intent(out) :: ierr
        !! Error code (0 if weights created, 1 otherwise)

        real(kind=GP) :: rescale_factor

        if(npoints == 1) then
            ! In the case of one point, no integration is performed.
            points(1)  = 1.0_GP
            weights(1) = 1.0_GP
            ierr = 0
        else if(npoints > 64) then
            ! In the case of more than 64 points we return an error.
            ! NOTE: This is the amount of points where the subroutine that
            !       creates the Gauss-Laguerre points has been tested for
            !       sufficient accuracy in PARALLAX.
            ierr = 1
        else
            call gauss_laguerre(points, weights)

            ! We rescale the points and weights to account for the fact that
            ! we want to specify the domain boundaries that should be used
            ! instead of using the optimal domain given by the Gauss points
            ! themselves.
            weights        = weights * exp(points)
            rescale_factor = length  / sum(weights)
            points         = points  * rescale_factor
            weights        = weights * rescale_factor

            ierr = 0
        end if

    end subroutine

end module
