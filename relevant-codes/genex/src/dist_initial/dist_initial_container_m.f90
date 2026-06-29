module dist_initial_container_m
    !! Module containing different initial distribution functions for the
    !! initialization of the simulation

    use genex_fortran_env_m, only : GP
    use dist_initial_m, only: dist_initial_t

    implicit none
    private

    type, public :: dist_initial_container_t
        !! Type that contains a collection of initial distributions
        class(dist_initial_t), private, allocatable :: dist_initial
        !! Initial distribution
    contains
        procedure, public :: initialize
        !! Initializes the initial distribution container
        procedure, public :: eval
        !! Evaluates the initial dist func for given real space coordinates
    end type

contains

    subroutine initialize(this, dist_initial)
        class(dist_initial_container_t), intent(inout) :: this
        !! Instance of the type
        class(dist_initial_t), intent(in) :: dist_initial
        !! Initial distribution type to initialize this with

        this%dist_initial = dist_initial
    end subroutine

    function eval(this, rho, theta, phi, vp, muB) result(res)
        class(dist_initial_container_t), intent(in) :: this
        !! Instance of the type
        real(kind = GP), intent(in) :: rho
        !! Normalized poloidal flux surface label
        real(kind = GP), intent(in) :: theta
        !! Poloidal angle
        real(kind = GP), intent(in) :: phi
        !! Toroidal angle
        real(kind = GP), intent(in) :: vp
        !! Parallel velocity
        real(kind = GP), intent(in) :: muB
        !! Magnetic moment times absolute magnetic field

        real(kind = GP) :: res
        !! Result

        res = this%dist_initial%eval(rho, theta, phi, vp, muB)
    end function
end module
