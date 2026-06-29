module bsg_operators_m
    !! Contains special operators that work with the BSG
    use genex_fortran_env_m, only: GP
    use mesh_5d_m, only: mesh_5d_t
    use bsg_types_m, only: vspace_bsg_t
    use bsg_interpolation_m, only: bsg_interpolation_t

    implicit none

    type, public :: bsg_operators_t
        !! Derived type for all BSG defined operators
        type(mesh_5d_t), private, pointer :: mesh
        !! Mesh
        class(bsg_interpolation_t), allocatable, private :: interpolator
        !! Interpolation instance
        integer, private :: vp_size
        !! Size of vp grid
        integer, contiguous, pointer, private, dimension(:) :: bsg_flags
        !! BSG flags
        type(vspace_bsg_t), pointer, private, dimension(:) :: vp_bsg_ptr
        !! Vp_grid pointer for BSG
        type(vspace_bsg_t), pointer, private, dimension(:) :: vpw_bsg_ptr
        !! V_grid weights pointer for BSG
        integer, private, dimension(:,:,:), allocatable :: local_start_idx
        !! Local start index for each block of the vp array for interpolation
        integer, private, dimension(:,:,:), allocatable :: local_end_idx
        !! Local end index for each block of the vp array for interpolation
        real(kind=GP), private, dimension(:,:,:,:), allocatable :: &
                                                             interpolation_coeff
        !! Interpolation coefficients
        logical, private, dimension(:,:,:), allocatable :: is_internal
        !! Specifies if a point on the block boundary is internal or external
        real(kind=GP), private, dimension(:), allocatable :: vp_delta
        !! Grid spacing for vp_grid for all blocks
        real(kind=GP), private, dimension(:), allocatable :: vp_delta_inverse
        !! 1/delta_vp for all blocks
        logical, private :: initialized = .false.
        !! Indicates if the type is initialized
    contains
        procedure, public  :: initialize
        procedure, public  :: interpolate_dist
        procedure, public  :: dot_product_bsg
        procedure, private :: initialize_vp_pointers

        procedure, public  :: get_local_start_idx_pointer
        procedure, public  :: get_local_end_idx_pointer
        procedure, public  :: get_is_internal_pointer
        procedure, public  :: get_vp_pointer
        procedure, public  :: get_vpw_pointer
        procedure, public  :: get_vp_delta_pointer
        procedure, public  :: get_prefac_vp_pointer
        procedure, public  :: is_initialized

    end type bsg_operators_t

    interface
        module subroutine initialize(this, mesh, num_blocks, interp_order)
            !! Initializes the BSG operators type
            class(bsg_operators_t), intent(inout) :: this
            !! Instance of the type
            type(mesh_5d_t), target, intent(in) :: mesh
            !! Mesh instance
            integer, intent(in) :: num_blocks
            !! Number of blocks
            integer, optional, intent(in) :: interp_order
            !! Interpolation order
        end subroutine

        module subroutine initialize_vp_pointers(this, block_number, &
                                               ptr_from_mesh, ptr_accessor)
            !! Initializes the velocity space grid pointer accessors
            class(bsg_operators_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: block_number
            !! Block number
            real(kind=GP), dimension(:), target, intent(in) :: ptr_from_mesh
            !! Pointer from mesh
            type(vspace_bsg_t), dimension(:), pointer, intent(inout) :: &
                                                                    ptr_accessor
            !! Pointer accessor
        end subroutine

        module subroutine interpolate_dist(this, nb_point, nb_neigh, i_neigh, &
                                           k, l, f_in, lb1, lb2, lb3, f_interp)
            !! Updates the distribution function value of neighbor points
            !! through interpolation
            class(bsg_operators_t), intent(inout) :: this
            !! Instance of type
            integer, intent(in) :: nb_point
            !! Block number of RZ point
            integer, intent(in) :: nb_neigh
            !! Block number of neighbor point
            integer, intent(in) :: i_neigh
            !! Neighbor point index
            integer, intent(in) :: k
            !! Index for phi dimension of the dist function
            integer, intent(in) :: l
            !! Index for vp dimension of the dist function
            integer, intent(in) :: lb1
            !! Lower bound of RZ dimension
            integer, intent(in) :: lb2
            !! Lower bound of phi dimension
            integer, intent(in) :: lb3
            !! Lower bound of vp dimension
            real(kind=GP), dimension(lb1:,lb2:,lb3:), intent(in) :: f_in
            !! Input distribution function
            real(kind=GP), intent(inout) :: f_interp
            !! Final interpolated value
        end subroutine

        module subroutine dot_product_bsg(this, coeff, points_neigh, lb1, lb2, &
                                          lb3, f_in, nb_point, k, l, f_dot)
            !! Performs the dot product for neighbors situated in different
            !! blocks
            class(bsg_operators_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:), intent(in) :: coeff
            !! Coefficient vector
            integer, dimension(:), intent(in) :: points_neigh
            !! Indices i of RZ dimension of all neighbor points
            integer, intent(in) :: lb1
            !! Lower bound of RZ dimension
            integer, intent(in) :: lb2
            !! Lower bound of phi dimension
            integer, intent(in) :: lb3
            !! Lower bound of vp dimension
            real(kind=GP), dimension(lb1:,lb2:,lb3:), intent(in) :: f_in
            !! Input distribution function
            integer, intent(in) :: nb_point
            !! Block number of RZ point
            integer, intent(in) :: k
            !! Index for phi dimension of the dist function
            integer, intent(in) :: l
            !! Index for vp dimension of the dist function
            real(kind=GP), intent(out) :: f_dot
            !! Final dot product value
        end subroutine
    end interface

contains

    function get_vp_pointer(this) result(ptr)
        !! Returns a pointer to the vp array
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vp_bsg_ptr
    end function

    function get_vpw_pointer(this) result(ptr)
        !! Returns a pointer to the vpw array
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vpw_bsg_ptr
    end function

    function get_local_start_idx_pointer(this) result(ptr)
        !! Returns a pointer to the local start index array
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, dimension(:,:,:), pointer :: ptr

        ptr => this%local_start_idx
    end function

    function get_local_end_idx_pointer(this) result(ptr)
        !! Returns a pointer to the local end index array
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, dimension(:,:,:), pointer :: ptr

        ptr => this%local_end_idx
    end function

    function get_is_internal_pointer(this) result(ptr)
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        logical, contiguous, dimension(:,:,:), pointer :: ptr

        ptr => this%is_internal
    end function

    function get_vp_delta_pointer(this) result(ptr)
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, dimension(:), pointer :: ptr

        ptr => this%vp_delta
    end function

    function get_prefac_vp_pointer(this) result(ptr)
        class(bsg_operators_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, dimension(:), pointer :: ptr

        ptr => this%vp_delta_inverse
    end function

    pure logical function is_initialized(this)
        class(bsg_operators_t), intent(in) :: this
        !! Instance of the type

        is_initialized = this%initialized
    end function

end module
