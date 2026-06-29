module block_structured_grids_m
    !! Module containing the implementation of block-structured grids (BSG).
    use genex_fortran_env_m, only: GP

    ! From PARALLAX
    use equilibrium_m, only: equilibrium_t
    use csrmat_m, only: csrmat_t => csrmat_t

    implicit none

    type, public :: block_structured_grids_t
        !! Type for the implementation of block-structured grids.
        integer, private :: num_blocks
        !! Number of blocks in the RZ plane
        logical, private :: initialized = .false.
        !! Indicates if it is initialized
        real(kind=GP), dimension(:), allocatable :: radial_markers
        !! Radial locations of the block boundaries
        real(kind=GP), private, dimension(:), allocatable :: vp_length_blocks
        !! Length of vp grid for each block
        integer, private :: vp_size
        !! Number of vp points
        real(kind=GP), private :: vp_delta_sg
        !! Grid spacing of a structured vp grid with uniform spacing
        real(kind=GP), private :: vp_length_total
        !! Length of total vp domain
        integer, private, allocatable, dimension(:) :: bsg_flags
        !! Flags for RZ point based on the block creation
    contains
        procedure, public :: initialize
        procedure, public :: construct
        procedure, public :: set_flags
        procedure, public :: set_map_matrix_flag
        procedure, public :: init_vp_grid

        procedure, public :: get_num_blocks
        procedure, public :: is_initialized
        procedure, public :: get_vp_length
        procedure, public :: get_bsg_flags_pointer
        procedure, public :: get_radial_markers

    end type block_structured_grids_t

    interface
        module subroutine initialize(this, num_blocks, radial_markers)
            !! Initialize the block-structured grids
            class(block_structured_grids_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: num_blocks
            !! Number of blocks in the RZ plane
            real(kind=GP), dimension(:), optional, intent(in) :: radial_markers
            !! Radial locations of the block boundaries
        end subroutine

        module subroutine construct(this, mesh_r, mesh_z, is_core, &
                                    equilibrium_instance)
            !! Construct blocks based on the RZ grid
            class(block_structured_grids_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:), target, intent(in) :: mesh_r, mesh_z
            !! Coordinate values of RZ grid
            real(kind=GP), dimension(:), optional, intent(in) :: is_core
            !! Mask to decide rho dependency for block creation
            class(equilibrium_t), optional, intent(in) :: equilibrium_instance
            !! Instance of the equilibrium
        end subroutine

        module subroutine set_flags(this, map_p, map_pp, map_m, map_mm, &
                                    neighbor_indices)
            !! Set flags for the map matrices and adjust bflag based on neighbor
            !! indices
            class(block_structured_grids_t), intent(inout) :: this
            !! Instance of the type
            type(csrmat_t), intent(in) :: map_p, map_pp, map_m, map_mm
            !! Map matrices for each point
            integer, dimension(:,:,:), intent(in) :: neighbor_indices
            !! neighboring indices
        end subroutine

        integer module function set_map_matrix_flag(this, bflag_rz, bflag_array)
            !! Set the flag for the map matrix
            class(block_structured_grids_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: bflag_rz
            !! bflag index of the RZ point
            integer, dimension(:), intent(in) :: bflag_array
            !! Array of bflag indices of the interpolation matrix
        end function

        module subroutine init_vp_grid(this, vp_size, vp_length_total, &
                                       vplen_markers)
            !! Load Structured vp_grid information.
            class(block_structured_grids_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: vp_size
            !! Number of vp grid points
            real(kind=GP), intent(in) :: vp_length_total
            !! Length of total vp domain
            real(kind=GP), dimension(:), intent(in) :: vplen_markers
            !! Length of vp space for each block
        end subroutine
    end interface

contains

    pure integer function get_num_blocks(this)
      !! Returns number of blocks
      class(block_structured_grids_t), intent(in) :: this
      !! Instance of the type
      get_num_blocks = this%num_blocks
    end function

    function get_bsg_flags_pointer(this) result(ptr)
        !! Returns a pointer of BSG_flags.
        class(block_structured_grids_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:) :: ptr

        ptr => this%bsg_flags
    end function

    pure real(kind=GP) function get_vp_length(this, block_number)
        !! Returns the vp length based on the block number for BSG
        class(block_structured_grids_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: block_number

        get_vp_length = this%vp_length_blocks(block_number)
    end function

    pure logical function is_initialized(this)
        !! Returns if BSG class is initialized
        class(block_structured_grids_t), intent(in) :: this
        !! Instance of the type
        is_initialized = this%initialized
    end function

    pure real(kind=GP) function get_radial_markers(this, block_number)
        !! Returns the radial markers based on the block number
        class(block_structured_grids_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: block_number

        get_radial_markers = this%radial_markers(block_number)
    end function

end module
