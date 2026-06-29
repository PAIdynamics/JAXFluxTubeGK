module structured_grids_m
    !! Module containing the implementation of the structured grids. This
    !! includes an abstract base class as well as the vp, mu and phi grids.
    use genex_fortran_env_m, only: GP

    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_MESH

    implicit none

    type, public, abstract :: structured_grid_t
        !! Abstract base type for the implementation of structured grids
        integer, private :: number_of_points
        !! Number of elements in the grid
        real(kind=GP), private, dimension(:), allocatable :: points
        !! Value of the grid
        real(kind=GP), private, dimension(:), allocatable :: weights
        !! Weights for numerical integration
        character(len=24) :: grid_type
        !! Type of the underlying grid
        character(len=24) :: quad_type
        !! Type of the quadrature used
        logical :: is_vspectral
        !! Indicates if spectral velocity space is used
    contains
        procedure, public :: get_pointer
        procedure, public :: get_weights_pointer
        procedure, public :: get_size
        procedure, public :: get_spacing
        procedure, public :: get_grid_type
        procedure, public :: get_quad_type
        procedure, public :: get_is_vspectral

    end type structured_grid_t

    type, public, extends(structured_grid_t) :: vp_grid_t
        !! Type implementing the grid in vp direction. This grid is symmetric.
    contains
        procedure, public :: initialize => vp_initialize
        procedure, public :: construct => vp_construct
    end type vp_grid_t

    interface
        module subroutine vp_initialize(this, npoints, is_vspectral)
            !! Initializes the grid in the vp direction
            class(vp_grid_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: npoints
            !! Number of points in vp direction
            logical, optional, intent(in) :: is_vspectral
            !! Spectral velocity switch
        end subroutine
        module subroutine vp_construct(this, length, quad_type)
            !! Initializes the grid in the vp direction
            class(vp_grid_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: length
            !! One-sided length in vp direction
            character(len=*), intent(in) :: quad_type
            !! Quadrature type
        end subroutine
    end interface

    type, public, extends(structured_grid_t) :: mu_grid_t
        !! Type implementing the grid in mu direction. This grid is asymmetric
        !! and covers only one half-plane around the origin.
        real(kind=GP), private, dimension(:), allocatable :: sqrt_points
        !! Value of the sqrt grid
    contains
        procedure, public :: initialize => mu_initialize
        procedure, public :: get_sqrt_pointer => mu_get_sqrt_pointer
        procedure, public :: get_sqrt_spacing => mu_get_sqrt_spacing
    end type mu_grid_t

    interface
        module subroutine mu_initialize(this, length, npoints, grid_type, &
                                        is_vspectral)
            !! Initializes the grid in the mu direction
            class(mu_grid_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: length
            !! Length in mu direction
            integer, intent(in) :: npoints
            !! Number of points in mu direction
            character(len=*), intent(in) :: grid_type
            !! Grid type
            logical, optional, intent(in) :: is_vspectral
            !! Spectral velspace switch (False by default)
        end subroutine

        module function mu_get_sqrt_pointer(this) result(ptr)
            !! Returns a pointer of rank 1 pointing to the sqrt grid valued
            !! stored in points.
            class(mu_grid_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), pointer, contiguous, dimension(:) :: ptr
        end function

        pure module real(kind=GP) function mu_get_sqrt_spacing(this, i)
            !! Returns the sqrt grid spacing at a given grid index i.
            !! If index is not valid, returns zero.
            class(mu_grid_t), target, intent(in) :: this
            !! Instance of the type
            integer, intent(in) :: i
            !! Grid location to fetch spacing from
        end function
    end interface

    type, public, extends(structured_grid_t) :: phi_grid_t
        !! Type implementing the grid in phi direction
    contains
        procedure, public :: initialize => phi_initialize
    end type phi_grid_t

    interface
        module subroutine phi_initialize(this, length, npoints)
            !! Initializes the grid in the phi direction. This is a 2pi
            !! periodic grid.
            class(phi_grid_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: length
            !! Length in phi direction (in most cases this should be 2 * PI)
            integer, intent(in) :: npoints
            !! Number of points in phi direction
        end subroutine
    end interface

contains

    function get_pointer(this) result(ptr)
        !! Returns a pointer of rank 1 pointing to the grid valued stored in
        !! points.
        class(structured_grid_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, contiguous, dimension(:) :: ptr

        ptr => this%points
    end function

    function get_weights_pointer(this) result(ptr)
        !! Returns a pointer of rank 1 pointing to the integration weights
        !! stored in the weights array
        class(structured_grid_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, contiguous, dimension(:) :: ptr

        ptr => this%weights
    end function

    pure integer function get_size(this)
        !! Returns the number of grid points
        class(structured_grid_t), intent(in) :: this
        !! Instance of the type

        get_size = size(this%points)
    end function

    pure character(len=24) function get_grid_type(this)
        !! Returns the grid type
        class(structured_grid_t), intent(in) :: this
        !! Instance of the type
        get_grid_type = this%grid_type
    end function

    pure character(len=24) function get_quad_type(this)
        !! Returns the quadrature type
        class(structured_grid_t), intent(in) :: this
        !! Instance of the type
        get_quad_type = this%quad_type
    end function

    pure logical function get_is_vspectral(this)
        !! Returns boolean if use spectral velspace
        class(structured_grid_t), intent(in) :: this
        !! Instance of the type
        get_is_vspectral = this%is_vspectral
    end function

    pure real(kind=GP) function get_spacing(this, i)
        !! Returns the grid spacing at a given grid index i. If index is not
        !! valid, returns zero.
        class(structured_grid_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: i
        !! Grid location to fetch spacing from

        if (i >= 1 .and. i < this%number_of_points) then
            get_spacing = this%points(i + 1) - this%points(i)
        else if (this%number_of_points == 1) then
            ! Special case only 1 point in grid
            get_spacing = this%points(1)
        else
            get_spacing = 0.0_GP
        end if
    end function

end module structured_grids_m
