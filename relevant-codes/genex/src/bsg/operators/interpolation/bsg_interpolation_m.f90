module bsg_interpolation_m
    !! Module containing the implementation of BSG interpolations.
    use genex_fortran_env_m, only: GP

    implicit none

    type, public, abstract :: bsg_interpolation_t
        !! Abstract base type for the implementation of BSG interpolation.
        integer, private :: interp_order
        !! Order of interpolation
    contains
        procedure, public :: initialize
        procedure, public :: eval_lsp_equidistant
        procedure(eval_local_index_interface), deferred :: eval_local_index
        procedure(eval_coeff_interface), deferred :: eval_coeff
        procedure(interpolate_interface), deferred :: interpolate
        procedure, public :: get_interpolation_order
    end type bsg_interpolation_t

    interface
        module subroutine initialize(this, interp_order)
            !! Initializes the interpolation class
            class(bsg_interpolation_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: interp_order
            !! Order of interpolation
        end subroutine

        integer module function eval_lsp_equidistant(this, x, xlb, x1, dx)
            !! Evaluates the local starting point for equidistant grids
            class(bsg_interpolation_t), intent(in) :: this
            !! Instanse of the type
            real(kind=GP), intent(in) :: x
            !! Location of interpolation
            integer, intent(in) :: xlb
            !! Lower bound of x array
            real(kind=GP), intent(in) :: x1
            !! Value of x at lower bound
            real(kind=GP), intent(in) :: dx
            !! Grid spacing of the x array
        end function

        module subroutine eval_local_index_interface(this, x, x_values, &
                                                     start_idx, end_idx)
            !! Evaluates the local start and end indices for interpolation
            class(bsg_interpolation_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: x
            !! Location of interpolation point
            real(kind=GP), dimension(:), intent(in) :: x_values
            !! Locations of data points
            integer, intent(out) :: start_idx
            !! Start index of xarray required for interpolation
            integer, intent(out) :: end_idx
            !! End index of xarray required for interpolation
        end subroutine

        module subroutine interpolate_interface(this, x_values, y_values, &
                                                coeff, x, y_interp)
            !! Performs interpolation
            class(bsg_interpolation_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:), intent(in) :: x_values
            !! Locations of data points
            real(kind=GP), dimension(:), intent(in) :: y_values
            !! Values at the data points
            real(kind=GP), dimension(:), intent(in) :: coeff
            !! Coefficients of the interpolation
            real(kind=GP), intent(in) :: x
            !! Location of interpolation point
            real(kind=GP), intent(out) :: y_interp
            !! Interpolated value at x
        end subroutine

        module subroutine eval_coeff_interface(this, x_values, coeff)
            !! Evaluates the coefficients for interpolation
            class(bsg_interpolation_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:), intent(in) :: x_values
            !! Locations of data points
            real(kind=GP), dimension(:), intent(out) :: coeff
            !! Coefficients of the interpolation
        end subroutine
    end interface

    type, public, extends(bsg_interpolation_t) :: equidistant_interpolation_t
        !! Type implementing the equidistant interpolation
    contains
        procedure, public :: eval_local_index => eval_local_index_equidistant
        procedure, public :: interpolate => interpolate_equidistant
        procedure, public :: eval_coeff => eval_coeff_equidistant
    end type equidistant_interpolation_t

    interface
        module subroutine eval_local_index_equidistant(this, x, x_values, &
                                                       start_idx, end_idx)
            class(equidistant_interpolation_t), intent(inout) :: this
            real(kind=GP), intent(in) :: x
            real(kind=GP), dimension(:), intent(in) :: x_values
            integer, intent(out) :: start_idx
            integer, intent(out) :: end_idx
        end subroutine

        module subroutine interpolate_equidistant(this, x_values, &
                                                  y_values, coeff, x, y_interp)
            class(equidistant_interpolation_t), intent(inout) :: this
            real(kind=GP), dimension(:), intent(in) :: x_values
            real(kind=GP), dimension(:), intent(in) :: y_values
            real(kind=GP), dimension(:), intent(in) :: coeff
            real(kind=GP), intent(in) :: x
            real(kind=GP), intent(out) :: y_interp
        end subroutine

        module subroutine eval_coeff_equidistant(this, x_values, coeff)
            class(equidistant_interpolation_t), intent(inout) :: this
            real(kind=GP), dimension(:), intent(in) :: x_values
            real(kind=GP), dimension(:), intent(out) :: coeff
        end subroutine
    end interface

contains

    pure integer function get_interpolation_order(this)
        !! Returns the order of interpolation
        class(bsg_interpolation_t), intent(in) :: this
        !! Instance of the type

        get_interpolation_order = this%interp_order
    end function

end module
