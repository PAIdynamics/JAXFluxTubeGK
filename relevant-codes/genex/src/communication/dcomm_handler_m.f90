module dcomm_handler_m
    !! Module containing the definition of the dcomm_handler_t type and
    !! interface
    use, intrinsic :: iso_fortran_env, only: INT64
    use, intrinsic :: iso_c_binding, only: C_PTR

    implicit none

    integer, private, parameter :: n_dims = 5
    !! Number of MPI parallelized dimensions

    type, public :: dcomm_handler_t
        !! Type handling the virtual domain decomposition (d) and the MPI
        !! communication (comm)
        logical, private :: is_initialized_comm = .false.
        !! Switch to check whether the communication part has been initialized
        logical, private :: is_initialized_domain = .false.
        !! Switch to check whether the domain decomposition part has been
        !! initialized
#ifdef ENABLE_GPU
        type(C_PTR), private :: dcomm_handler_cxx_pptr
        !! C pointer to the dcomm_handler_t C++ class instance pointer
#endif

        ! Communication part

        integer, private :: comm_cart
        !! MPI communicator for the complete topology
        integer, private :: comm_phi
        !! MPI communicator for the phi direction
        integer, private :: comm_vp
        !! MPI communicator for the vp direction
        integer, private :: comm_mu
        !! MPI communicator for the mu direction
        integer, private :: comm_sp
        !! MPI communicator for the sp direction
        integer, private :: comm_phi_sp
        !! MPI communicator for the phi and sp direction
        integer, private :: comm_vp_mu
        !! MPI communicator for the vp and mu direction
        integer, private :: comm_phi_vp_mu
        !! MPI communicator for the phi, vp and mu direction
        integer, private :: comm_vp_mu_sp
        !! MPI communicator for the vp and mu and sp direction
        integer, private :: comm_mu_sp
        !! MPI communicator for mu and sp direction
        integer, private :: comm_phi_mu_sp
        !! MPI communicator for phi, mu and sp direction
        integer, private :: n_procs_total
        !! Number of procs in total
        integer, private :: n_procs_RZ
        !! Number of procs in RZ direction
        integer, private :: n_procs_phi
        !! Number of procs in phi direction
        integer, private :: n_procs_mu
        !! Number of procs in mu direction
        integer, private :: n_procs_vp
        !! Number of procs in vp direction
        integer, private :: n_procs_sp
        !! Number of procs in sp direction
        integer, private :: rank
        !! Number of the rank the code is running on
        integer, private :: cart_coords(n_dims)
        !! Coordinates in the Cartesian topology

        ! Global problem size

        integer, private :: n_points_global_RZ
        !! Global number of points in RZ
        integer, private :: n_points_global_phi
        !! Global number of points in phi
        integer, private :: n_points_global_vp
        !! Global number of points in vp
        integer, private :: n_points_global_mu
        !! Global number of points in mu
        integer, private :: n_points_global_sp
        !! Global number of points in sp

        ! Domain decomposition part

        integer, private, dimension(n_dims) :: dim_permut
        !! Specifies the order of the dimensions
        integer, private, dimension(n_dims) :: number_of_data_elements
        !! Number of data elements for each dimension (excludes ghost cells)
        integer, private, dimension(n_dims) :: number_of_elements
        !! Number of elements for each dimension (includes ghost cells)
        integer, private, dimension(n_dims) :: number_of_ghosts
        !! Number of ghost cells for each dimension
        integer, private, dimension(n_dims) :: lb
        !! Lower boundary index of each dimension
        integer, private, dimension(n_dims) :: ub
        !! Upper boundary index of each dimension
        integer, private, dimension(n_dims) :: lb_stripped
        !! Lower boundary index without ghosts for each dimension
        integer, private, dimension(n_dims) :: ub_stripped
        !! Upper boundary index without ghosts for each dimension
    contains
        procedure, public  :: initialize => initialize_dcomm_handler
        procedure, private :: initialize_communication
        procedure, private :: initialize_pvm_domain
        procedure, public  :: initialize_RZ_domain
        procedure, private :: check_parallelization
        final :: finalize

        procedure, public :: print_mesh_info

        procedure, public :: get_n_procs_RZ
        procedure, public :: get_n_procs_phi
        procedure, public :: get_n_procs_vp
        procedure, public :: get_n_procs_mu
        procedure, public :: get_n_procs_sp
        procedure, public :: get_comm_cart
        procedure, public :: get_cart_coords
        procedure, public :: get_comm_phi
        procedure, public :: get_comm_phi_sp
        procedure, public :: get_comm_phi_vp_mu
        procedure, public :: get_comm_phi_mu_sp
        procedure, public :: get_comm_vp
        procedure, public :: get_comm_vp_mu
        procedure, public :: get_comm_vp_mu_sp
        procedure, public :: get_comm_mu
        procedure, public :: get_comm_mu_sp
        procedure, public :: get_comm_sp
        procedure, public :: get_rank
        procedure, public, nopass :: get_n_dims
        procedure, public :: is_master
        procedure, public :: is_initialized

        procedure, public :: get_size
        procedure, public :: get_size_stripped
        procedure, public :: get_dim_permut
        procedure, public :: get_number_of_data_elements
        procedure, public :: get_number_of_elements
        procedure, public :: get_number_of_ghosts
        procedure, private :: get_lbound_element
        procedure, private :: get_ubound_element
        procedure, private :: get_lbound_stripped_element
        procedure, private :: get_ubound_stripped_element
        procedure, private :: get_lbound_array
        procedure, private :: get_ubound_array
        procedure, private :: get_lbound_stripped_array
        procedure, private :: get_ubound_stripped_array
        generic, public :: get_lbound => get_lbound_element, get_lbound_array
        generic, public :: get_ubound => get_ubound_element, get_ubound_array
        generic, public :: get_lbound_stripped => get_lbound_stripped_element, &
                                                  get_lbound_stripped_array
        generic, public :: get_ubound_stripped => get_ubound_stripped_element, &
                                                  get_ubound_stripped_array

#ifdef ENABLE_GPU
        procedure, private :: initialize_gpu
        procedure, private :: finalize_gpu
        procedure, public :: get_cxx_pointer => get_dcomm_handler_cxx_pointer
#endif
    end type

    interface
        module subroutine initialize_dcomm_handler(this, comm, &
                                                   n_procs_phi, n_procs_vp, &
                                                   n_procs_mu, n_procs_sp, &
                                                   dim_permut)
            !! Initializes the type by creating the required MPI communicators
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: comm
            !! Communicator to create topology from
            integer, intent(in) :: n_procs_phi
            !! Number of processes in phi direction
            integer, intent(in) :: n_procs_vp
            !! Number of processes in vp direction
            integer, intent(in) :: n_procs_mu
            !! Number of processes in mu direction
            integer, intent(in) :: n_procs_sp
            !! Number of processes in sp direction
            integer, dimension(:), optional, intent(in) :: dim_permut
            !! Order of dimensions used. Default = [1, 2, 3, 4, 5]
        end subroutine

        module subroutine initialize_communication(this, comm, &
                                                   n_procs_phi, n_procs_vp, &
                                                   n_procs_mu, n_procs_sp)
            !! Initializes the type by creating the required MPI communicators
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: comm
            !! Base MPI communicator where to create the topology from
            integer, intent(in) :: n_procs_phi
            !! Number of processes in phi direction
            integer, intent(in) :: n_procs_vp
            !! Number of processes in vp direction
            integer, intent(in) :: n_procs_mu
            !! Number of processes in mu direction
            integer, intent(in) :: n_procs_sp
            !! Number of processes in sp direction
        end subroutine

        module subroutine check_parallelization(this, comm, n_procs_RZ, &
                                                n_procs_phi, n_procs_vp, &
                                                n_procs_mu, n_procs_sp)
            !! Checks the parallelization and initializes the number of procs
            !! components of the type.
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: comm
            !! Base MPI communicator where to create the topology from
            integer, intent(in) :: n_procs_RZ
            !! Number of processes in RZ direction
            integer, intent(in) :: n_procs_phi
            !! Number of processes in phi direction
            integer, intent(in) :: n_procs_mu
            !! Number of processes in mu direction
            integer, intent(in) :: n_procs_vp
            !! Number of processes in vp direction
            integer, intent(in) :: n_procs_sp
            !! Number of processes in sp direction
        end subroutine

        module subroutine initialize_gpu(this, comm)
            !! Initialize the dcomm_handler_t C++ class, including allocation
            !! on the device memory
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: comm
            !! Base MPI communicator where to create the topology from
        end subroutine

        module subroutine finalize_gpu(this)
            !! Finalize the dcomm_handler_t C++ class, including deallocation
            !! from the device memory
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_pvm_domain(this, dim_permut)
            !! Initializes the domain of the parallelized phi, vp and mu
            !! dimensions
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(:), intent(in) :: dim_permut
            !! Order of dimensions used
        end subroutine

        module subroutine initialize_RZ_domain(this, n_points_RZ)
            !! Initializes the domain of the not parallelized RZ dimension
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ
        end subroutine

        module subroutine finalize(this)
            !! Finalizes the object and release resources used in the
            !! communication
            type(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine print_mesh_info(this)
            !! Print global and local mesh info to debug channel
            class(dcomm_handler_t), intent(inout) :: this
            !! Instance of the type
        end subroutine
    end interface

contains

    pure integer function get_n_procs_RZ(this)
        !! Returns the number of MPI processes in RZ direction
        class(dcomm_handler_t), intent(in) :: this

        get_n_procs_RZ = this%n_procs_RZ
    end function

    pure integer function get_n_procs_phi(this)
        !! Returns the number of MPI processes in phi direction
        class(dcomm_handler_t), intent(in) :: this

        get_n_procs_phi = this%n_procs_phi
    end function

    pure integer function get_n_procs_vp(this)
        !! Returns the number of MPI processes in vp direction
        class(dcomm_handler_t), intent(in) :: this

        get_n_procs_vp = this%n_procs_vp
    end function

    pure integer function get_n_procs_mu(this)
        !! Returns the number of MPI processes in mu direction
        class(dcomm_handler_t), intent(in) :: this

        get_n_procs_mu = this%n_procs_mu
    end function

    pure integer function get_n_procs_sp(this)
        !! Returns the number of MPI processes in sp direction
        class(dcomm_handler_t), intent(in) :: this

        get_n_procs_sp = this%n_procs_sp
    end function

#ifdef ENABLE_GPU
    function get_dcomm_handler_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the dcomm_handler_t C++ class instance
        !! pointer
        class(dcomm_handler_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%dcomm_handler_cxx_pptr
    end function
#endif

    pure function get_cart_coords(this) result(coords)
        !! Returns the coordinates of the cartesian topology
        class(dcomm_handler_t), intent(in) :: this
        integer, dimension(n_dims) :: coords
        coords = this%cart_coords
    end function

    pure integer function get_comm_cart(this)
        !! Returns the communicator containing the entire cartesian topology
        class(dcomm_handler_t), intent(in) :: this
        get_comm_cart = this%comm_cart
    end function

    pure integer function get_comm_phi(this)
        !! Returns the communicator in phi direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_phi = this%comm_phi
    end function

    pure integer function get_comm_phi_sp(this)
        !! Returns the communicator in phi and sp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_phi_sp = this%comm_phi_sp
    end function

    pure integer function get_comm_phi_vp_mu(this)
        !! Returns the communicator in phi, vp and mu direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_phi_vp_mu = this%comm_phi_vp_mu
    end function

    pure integer function get_comm_phi_mu_sp(this)
        !! Returns the communicator in phi, mu and sp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_phi_mu_sp = this%comm_phi_mu_sp
    end function

    pure integer function get_comm_vp(this)
        !! Returns the communicator in vp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_vp = this%comm_vp
    end function

    pure integer function get_comm_vp_mu(this)
        !! Returns the communicator in vp and mu direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_vp_mu = this%comm_vp_mu
    end function

    pure integer function get_comm_vp_mu_sp(this)
        !! Returns the communicator in vp, mu and sp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_vp_mu_sp = this%comm_vp_mu_sp
    end function

    pure integer function get_comm_mu(this)
        !! Returns the communicator in mu direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_mu = this%comm_mu
    end function

    pure integer function get_comm_mu_sp(this)
        !! Returns the communicator in mu and sp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_mu_sp = this%comm_mu_sp
    end function

    pure integer function get_comm_sp(this)
        !! Returns the communicator in sp direction
        class(dcomm_handler_t), intent(in) :: this
        get_comm_sp = this%comm_sp
    end function

    pure integer function get_rank(this)
        !! Returns the current rank
        class(dcomm_handler_t), intent(in) :: this
        get_rank = this%rank
    end function

    pure integer function get_n_dims()
        !! Returns the number of parallelized dimensions
        get_n_dims = n_dims
    end function

    pure logical function is_master(this)
        !! Checks whether the current rank is the master/0 rank
        class(dcomm_handler_t), intent(in) :: this
        is_master = (this%rank == 0)
    end function

    pure logical function is_initialized(this)
        !! Checks whether the type has been fully initialized
        class(dcomm_handler_t), intent(in) :: this
        is_initialized = this%is_initialized_comm &
                       .and. this%is_initialized_domain
    end function

    pure integer(kind=INT64) function get_size(this)
        !! Returns the size of the distributed domain with ghosts
        class(dcomm_handler_t), intent(in) :: this
        get_size = product(this%ub(1:n_dims) - this%lb(1:n_dims) + 1)
    end function

    pure integer(kind=INT64) function get_size_stripped(this)
        !! Returns the size of the distributed domain without ghosts
        class(dcomm_handler_t), intent(in) :: this
        get_size_stripped = product( this%ub_stripped(1:n_dims) &
                                   - this%lb_stripped(1:n_dims) + 1)
    end function

    pure integer function get_dim_permut(this, ind_dim)
        !! Returns the order of the dimensions
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_dim_permut = this%dim_permut(ind_dim)
    end function

    pure integer function get_number_of_data_elements(this, ind_dim)
        !! Returns the number of data elements for each dimension (excl. ghosts)
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_number_of_data_elements = this%number_of_data_elements(ind_dim)
    end function

    pure integer function get_number_of_elements(this, ind_dim)
        !! Returns the number of elements for each dimension (incl. ghosts)
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_number_of_elements = this%number_of_elements(ind_dim)
    end function

    pure integer function get_number_of_ghosts(this, ind_dim)
        !! Returns the number of ghosts for each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_number_of_ghosts = this%number_of_ghosts(ind_dim)
    end function

    pure integer function get_lbound_element(this, ind_dim)
        !! Returns the lower boundary index of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_lbound_element = this%lb(ind_dim)
    end function

    pure integer function get_ubound_element(this, ind_dim)
        !! Returns the upper boundary index of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_ubound_element = this%ub(ind_dim)
    end function

    pure integer function get_lbound_stripped_element(this, ind_dim)
        !! Returns the lower boundary index without ghosts of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_lbound_stripped_element = this%lb_stripped(ind_dim)
    end function

    pure integer function get_ubound_stripped_element(this, ind_dim)
        !! Returns the upper boundary index without ghosts of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer, intent(in) :: ind_dim
        get_ubound_stripped_element = this%ub_stripped(ind_dim)
    end function

    pure function get_lbound_array(this) result(lbound)
        !! Returns the lower boundary index of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer :: lbound(n_dims)
        lbound(1:n_dims) = this%lb(1:n_dims)
    end function

    pure function get_ubound_array(this) result(ubound)
        !! Returns the upper boundary index of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer :: ubound(n_dims)
        ubound(1:n_dims) = this%ub(1:n_dims)
    end function

    pure function get_lbound_stripped_array(this) result(lbound_s)
        !! Returns the lower boundary index without ghosts of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer :: lbound_s(n_dims)
        lbound_s(1:n_dims) = this%lb_stripped(1:n_dims)
    end function

    pure function get_ubound_stripped_array(this) result(ubound_s)
        !! Returns the upper boundary index without ghosts of each dimension
        class(dcomm_handler_t), intent(in) :: this
        integer :: ubound_s(n_dims)
        ubound_s(1:n_dims) = this%ub_stripped(1:n_dims)
    end function
end module
