module data_storage_m
    !! Module containing the definition and interface of the data storage
    !! types. The data storage includes storaging data_arrays with capabilities
    !! for MPI halos. These can be exchanged automatically by calling the
    !! start_exchange and finish_exchange procedure of the types.
    use, intrinsic :: iso_c_binding, only: C_PTR, C_INT32_T
    use genex_fortran_env_m, only: GP
    use dcomm_handler_m, only: dcomm_handler_t
    use mailbox_m, only: mailbox_t
    use data_array_m, only: data_array_5d_t, data_array_4d_t, data_array_2d_t

    implicit none

    type, public, abstract :: data_storage_t
        !! Type to store data for the state_vector
        type(dcomm_handler_t), pointer, private :: dcomm_handler
        !! MPI communication handler
        integer, private :: size
        !! Total number of elements in storage
        integer, private :: dimensions
        !! Number of dimensions of the stored data
        integer, private, dimension(:), allocatable :: number_of_elements
        !! Number of elements in storage for each dimension
        integer, private, dimension(:), allocatable :: lb
        !! Array specifying the lower boundary index of each dimension
        integer, private, dimension(:), allocatable :: ub
        !! Array specifying the upper boundary index of each dimension
        integer, private, dimension(:), allocatable :: lb_stripped
        !! Array specifying the lower boundary index without ghosts for each
        !! dimension
        integer, private, dimension(:), allocatable :: ub_stripped
        !! Array specifying the upper boundary index without ghosts for each
        !! dimension
        integer, private, dimension(:), allocatable :: number_of_ghost_cells
        !! Array specifying the number of ghost cells
        integer, private, dimension(:), allocatable :: number_of_data_cells
        !! Array specifying the number of data cells
        integer, private, dimension(:), allocatable :: number_of_mail_partners
        !! Number of mail partners for mpi ghost exchange
        integer, private, dimension(:), allocatable :: dim_permut
        !! Array specifying the order of the dimensions in storage.
        logical, private :: is_initialized
        !! Is .true. if initialize has been called, .false. otherwise
    contains
        procedure, private :: initialize_parent
        procedure, public :: get_size
        procedure, public :: get_dimension
        procedure, public :: get_num_mail_partners
    end type

    interface
        module subroutine initialize_parent(this, dcomm_handler, dimensions)
            !! Initializes the parent type
            class(data_storage_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            integer, intent(in) :: dimensions
            !! Number of dimensions
        end subroutine
    end interface

    type, public, abstract, extends(data_storage_t) :: data_storage_5d_t
        !! Type to store 5 dimensional fields with possible exchange in
        !! PHI, VP and MU dimensions
        class(data_array_5d_t), private, allocatable :: data_array
        !! Wrapper object for array of data storage and its bounds
        real(kind=GP), private, dimension(:,:,:,:,:), pointer :: storage
        !! Data storage
        type(mailbox_t), dimension(3) :: mailboxes
        !! Mailbox buffering data for boundary exchange in PHI, VP and MU
        !! dimension
        integer :: n_ex_dims = 3
        !! Number of exchange dimension, currently PHI, VP and MU exchange is
        !! possible
    contains
        procedure(initialize_5d), deferred, public :: initialize
        procedure(start_exchange_5d), deferred, public :: start_exchange
        procedure(finish_exchange_5d), deferred, public :: finish_exchange
        procedure, public :: get_data_pointer => get_data_array_pointer_5d
        procedure, public :: get_pointer => get_pointer_5d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_5d
        procedure, public :: shape => shape_5d
        procedure, public :: shape_stripped => shape_stripped_5d
        procedure, public :: lbound => lbound_5d
        procedure, public :: ubound => ubound_5d
        procedure, public :: lbound_stripped => lbound_stripped_5d
        procedure, public :: ubound_stripped => ubound_stripped_5d
        procedure, public :: update_host => update_host_5d
        procedure, public :: update_device => update_device_5d
        procedure, private :: get_mailbox_index => get_mailbox_index_5d
        procedure, private :: get_physical_dimension &
                              => get_physical_dimension_5d
        procedure, private :: has_edge_ghosts &
                              => has_edge_ghosts_5d
    end type

    interface

        module subroutine initialize_5d(this, dcomm_handler, init_value)
            !! Initializes the type
            class(data_storage_5d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            real(kind=GP), optional, intent(in) :: init_value
            !! Optional initial value to uniformly set to the array
        end subroutine

        module subroutine start_exchange_5d(this, ex_dim)
            !! Start the exchange of the ghost cells via mpi. This subroutine
            !! returns immediately. Other calculations can be performed while
            !! the exchange is ongoing.
            class(data_storage_5d_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: ex_dim
            !! Exchange dimension, PHI, VP and MU exchange is possible
        end subroutine

        module subroutine finish_exchange_5d(this, ex_dim)
            !! Finishes the exchange of the ghost cells via mpi. After this
            !! subroutine has returned it is guaranteed that the information
            !! of the ghost cells is present in the storage.
            class(data_storage_5d_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: ex_dim
            !! Exchange dimension, PHI, VP and MU exchange is possible
        end subroutine

        module function get_physical_dimension_5d(this, ex_dim) result(res)
            !! Returns the physical dimension associated to ex_dim via
            !! dim_permut. This corresponds to the inverse of dim_permut
            class(data_storage_5d_t), intent(in) :: this
            !! Instance of the type
            integer, intent(in) :: ex_dim
            !! Exchange dimension
            integer :: res
        end function

        module function get_mailbox_index_5d(this, ex_dim) result(mb_ind)
            !! Returns the index of the mailbox storage reserved for the
            !! exchange dimension
            class(data_storage_5d_t), intent(in) :: this
            !! Instance of the type
            integer, intent(in) :: ex_dim
            !! Exchange dimension
            integer :: mb_ind
        end function

        module function has_edge_ghosts_5d(this, ex_dim, comm_dim) result(res)
            !! Returns 1 if the ghost cells along the comm_dim dimensions
            !! are included in MPI exchange when exchanging along
            !! the ex_dim dimension. Required only for the spectral approach.
            !! Otherwise, returns 0.
            class(data_storage_5d_t), intent(in) :: this
            !! Instance of the type
            integer, intent(in) :: ex_dim
            !! Exchange dimension
            integer, intent(in) :: comm_dim
            !! Exchange dimension including the ghost cells
            integer :: res
        end function

    end interface

    type, public, extends(data_storage_5d_t) :: data_storage_cpu_5d_t
        !! Type to store 5 dimensional fields on the CPU
    contains
        procedure, public :: initialize => initialize_cpu_5d
        procedure, public :: start_exchange => start_exchange_cpu_5d
        procedure, public :: finish_exchange => finish_exchange_cpu_5d
    end type

    interface
        module subroutine initialize_cpu_5d(this, dcomm_handler, init_value)
            class(data_storage_cpu_5d_t), intent(inout) :: this
            type(dcomm_handler_t),  pointer, intent(in) :: dcomm_handler
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_cpu_5d(this, ex_dim)
            class(data_storage_cpu_5d_t), intent(inout) :: this
            integer, intent(in) :: ex_dim
        end subroutine

        module subroutine finish_exchange_cpu_5d(this, ex_dim)
            class(data_storage_cpu_5d_t), intent(inout) :: this
            integer, intent(in) :: ex_dim
        end subroutine

    end interface

    type, public, abstract, extends(data_storage_t) :: data_storage_4d_t
        !! Type to store 4 dimensional fields with exchange in
        !! the PHI dimension
        class(data_array_4d_t), private, allocatable :: data_array
        !! Wrapper object for array of data storage and its bounds
        real(kind=GP), private, dimension(:,:,:,:), pointer :: storage
        !! Data storage
        type(mailbox_t) :: mailbox
        !! Mailbox buffering data for boundary exchange in PHI dimension
    contains
        procedure(initialize_4d), deferred, public :: initialize
        procedure(start_exchange_4d), deferred, public :: start_exchange
        procedure(finish_exchange_4d), deferred, public :: finish_exchange
        procedure, public :: get_data_pointer => get_data_array_pointer_4d
        procedure, public :: get_pointer => get_pointer_4d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_4d
        procedure, public :: shape => shape_4d
        procedure, public :: shape_stripped => shape_stripped_4d
        procedure, public :: lbound => lbound_4d
        procedure, public :: ubound => ubound_4d
        procedure, public :: lbound_stripped => lbound_stripped_4d
        procedure, public :: ubound_stripped => ubound_stripped_4d
        procedure, public :: update_host => update_host_4d
        procedure, public :: update_device => update_device_4d
    end type

    interface
        module subroutine initialize_4d(this, dcomm_handler, &
                                        n_mom, n_spec, init_value)
            !! Initializes the type
            class(data_storage_4d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            integer, intent(in) :: n_mom
            !! Number of neutrals moments
            integer, intent(in) :: n_spec
            !! Number of neutrals species
            real(kind=GP), optional, intent(in) :: init_value
            !! Optional initial value to uniformly set to the array
        end subroutine

        module subroutine start_exchange_4d(this)
            !! Start the exchange of the ghost cells via mpi. This subroutine
            !! returns immediately. Other calculations can be performed while
            !! the exchange is ongoing.
            class(data_storage_4d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine finish_exchange_4d(this)
            !! Finishes the exchange of the ghost cells via mpi. After this
            !! subroutine has returned it is guaranteed that the information
            !! of the ghost cells is present in the storage.
            class(data_storage_4d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

    end interface

    type, public, extends(data_storage_4d_t) :: data_storage_cpu_4d_t
        !! Type to store 4 dimensional fields on the CPU
    contains
        procedure, public :: initialize => initialize_cpu_4d
        procedure, public :: start_exchange => start_exchange_cpu_4d
        procedure, public :: finish_exchange => finish_exchange_cpu_4d
    end type

    interface
        module subroutine initialize_cpu_4d(this, dcomm_handler, &
                                            n_mom, n_spec, init_value)
            class(data_storage_cpu_4d_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            integer, intent(in) :: n_mom
            integer, intent(in) :: n_spec
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_cpu_4d(this)
            class(data_storage_cpu_4d_t), intent(inout) :: this
        end subroutine

        module subroutine finish_exchange_cpu_4d(this)
            class(data_storage_cpu_4d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, abstract, extends(data_storage_t) :: data_storage_2d_t
        !! Type to store 2 dimensional fields with exchange in
        !! the PHI dimension
        class(data_array_2d_t), private, allocatable :: data_array
        !! Wrapper object for array of data storage and its bounds
        real(kind=GP), private, dimension(:,:), pointer :: storage
        !! Data storage
        type(mailbox_t) :: mailbox
        !! Mailbox buffering data for boundary exchange in PHI dimension
    contains
        procedure(initialize_2d), deferred, public :: initialize
        procedure(start_exchange_2d), deferred, public :: start_exchange
        procedure(finish_exchange_2d), deferred, public :: finish_exchange
        procedure, public :: get_data_pointer => get_data_array_pointer_2d
        procedure, public :: get_pointer => get_pointer_2d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_2d
        procedure, public :: shape => shape_2d
        procedure, public :: shape_stripped => shape_stripped_2d
        procedure, public :: lbound => lbound_2d
        procedure, public :: ubound => ubound_2d
        procedure, public :: lbound_stripped => lbound_stripped_2d
        procedure, public :: ubound_stripped => ubound_stripped_2d
        procedure, public :: update_host => update_host_2d
        procedure, public :: update_device => update_device_2d
    end type

    interface
        module subroutine initialize_2d(this, dcomm_handler, init_value)
            !! Initializes the type
            class(data_storage_2d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            real(kind=GP), optional, intent(in) :: init_value
            !! Optional initial value to uniformly set to the array
        end subroutine

        module subroutine start_exchange_2d(this)
            !! Start the exchange of the ghost cells via mpi. This subroutine
            !! returns immediately. Other calculations can be performed while
            !! the exchange is ongoing.
            class(data_storage_2d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine finish_exchange_2d(this)
            !! Finishes the exchange of the ghost cells via mpi. After this
            !! subroutine has returned it is guaranteed that the information
            !! of the ghost cells is present in the storage.
            class(data_storage_2d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

    end interface

    type, public, extends(data_storage_2d_t) :: data_storage_cpu_2d_t
        !! Type to store 2 dimensional fields on the CPU
    contains
        procedure, public :: initialize => initialize_cpu_2d
        procedure, public :: start_exchange => start_exchange_cpu_2d
        procedure, public :: finish_exchange => finish_exchange_cpu_2d
    end type

    interface
        module subroutine initialize_cpu_2d(this, dcomm_handler, init_value)
            class(data_storage_cpu_2d_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_cpu_2d(this)
            class(data_storage_cpu_2d_t), intent(inout) :: this
        end subroutine

        module subroutine finish_exchange_cpu_2d(this)
            class(data_storage_cpu_2d_t), intent(inout) :: this
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(data_storage_5d_t) :: data_storage_gpu_5d_t
        !! Type to store 5 dimensional fields on the GPU
        type(C_PTR), private :: data_storage_cxx_pptr
        !! C pointer to the data_storage_gpu_5d_t C++ class instance pointer
    contains
        procedure, public :: initialize => initialize_gpu_5d
        procedure, public :: start_exchange => start_exchange_gpu_5d
        procedure, public :: finish_exchange => finish_exchange_gpu_5d
        procedure, public :: get_cxx_pointer => get_data_storage_5d_cxx_pointer
        final :: finalize_gpu_5d
    end type

    interface
        module subroutine initialize_gpu_5d(this, dcomm_handler, init_value)
            class(data_storage_gpu_5d_t), intent(inout) :: this
            type(dcomm_handler_t),  pointer, intent(in) :: dcomm_handler
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_gpu_5d(this, ex_dim)
            class(data_storage_gpu_5d_t), intent(inout) :: this
            integer, intent(in) :: ex_dim
        end subroutine

        module subroutine finish_exchange_gpu_5d(this, ex_dim)
            class(data_storage_gpu_5d_t), intent(inout) :: this
            integer, intent(in) :: ex_dim
        end subroutine

        module subroutine finalize_gpu_5d(this)
            type(data_storage_gpu_5d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(data_storage_4d_t) :: data_storage_gpu_4d_t
        !! Type to store 4 dimensional fields on the GPU
        type(C_PTR), private :: data_storage_cxx_pptr
        !! C pointer to the data_storage_gpu_4d_t C++ class instance pointer
    contains
        procedure, public :: initialize => initialize_gpu_4d
        procedure, public :: start_exchange => start_exchange_gpu_4d
        procedure, public :: finish_exchange => finish_exchange_gpu_4d
        procedure, public :: get_cxx_pointer => get_data_storage_4d_cxx_pointer
        final :: finalize_gpu_4d
    end type

    interface
        module subroutine initialize_gpu_4d(this, dcomm_handler, &
                                            n_mom, n_spec, init_value)
            class(data_storage_gpu_4d_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            integer, intent(in) :: n_mom
            integer, intent(in) :: n_spec
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_gpu_4d(this)
            class(data_storage_gpu_4d_t), intent(inout) :: this
        end subroutine

        module subroutine finish_exchange_gpu_4d(this)
            class(data_storage_gpu_4d_t), intent(inout) :: this
        end subroutine

        module subroutine finalize_gpu_4d(this)
            type(data_storage_gpu_4d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(data_storage_2d_t) :: data_storage_gpu_2d_t
        !! Type to store 5 dimensional fields on the GPU
        type(C_PTR), private :: data_storage_cxx_pptr
        !! C pointer to the data_storage_gpu_2d_t C++ class instance pointer
    contains
        procedure, public :: initialize => initialize_gpu_2d
        procedure, public :: start_exchange => start_exchange_gpu_2d
        procedure, public :: finish_exchange => finish_exchange_gpu_2d
        procedure, public :: get_cxx_pointer => get_data_storage_2d_cxx_pointer
        final :: finalize_gpu_2d
    end type

    interface
        module subroutine initialize_gpu_2d(this, dcomm_handler, init_value)
            class(data_storage_gpu_2d_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            real(kind=GP), optional, intent(in) :: init_value
        end subroutine

        module subroutine start_exchange_gpu_2d(this)
            class(data_storage_gpu_2d_t), intent(inout) :: this
        end subroutine

        module subroutine finish_exchange_gpu_2d(this)
            class(data_storage_gpu_2d_t), intent(inout) :: this
        end subroutine

        module subroutine finalize_gpu_2d(this)
            type(data_storage_gpu_2d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, bind(C) :: data_storage_data_t
        !! Fortran/C++ interoperable structure of the class members
        !! relevant to data_storage_t
        integer(kind=C_INT32_T) :: array_dim
        !! Number of dimension or rank of the array
        integer(kind=C_INT32_T) :: n_ex_dims
        !! Number of echange dimension
        type(C_PTR) :: number_of_elements_ptr
        !! C pointer to the array specifying the number of elements of
        !! the stored data
        type(C_PTR) :: number_of_ghost_cells_ptr
        !! C pointer to the array specifying the number of ghost cells
        type(C_PTR) :: number_of_data_cells_ptr
        !! C pointer to the array specifying the number of data cells
        type(C_PTR) :: number_of_mail_partners_ptr
        !! C pointer to the array specifying the number of mail partners
        !! for mpi ghost exchange
        type(C_PTR) :: dim_permut_ptr
        !! C pointer to the array specifying the order of the dimensions
        !! in storage
    end type
#endif

contains

    function get_data_array_pointer_5d(this) result(ptr)
        !! Return a pointer of the data array class object containing 5d array
        class(data_storage_5d_t), target, intent(inout) :: this
        !! Instance of the type
        class(data_array_5d_t), pointer :: ptr

        ptr => this%data_array
    end function

    function get_pointer_5d(this) result(ptr)
        !! Returns a pointer of rank 5 pointing to the data stored in
        !! data_storage_5d_t
        class(data_storage_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: ptr

        ptr => this%storage
    end function

    function get_pointer_stripped_5d(this) result(ptr)
        !! Returns a pointer of rank 5 pointing to the data stored in
        !! data_storage_5d_cpu_t without ghost cells
        !! NOTE: The pointer is not contiguous anymore
        class(data_storage_5d_t), target, intent(inout) :: this
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: ptr

        ptr(this%lb_stripped(1):, this%lb_stripped(2):, this%lb_stripped(3):, &
            this%lb_stripped(4):, this%lb_stripped(5):) &
            => this%storage(this%lb_stripped(1):this%ub_stripped(1), &
                            this%lb_stripped(2):this%ub_stripped(2), &
                            this%lb_stripped(3):this%ub_stripped(3), &
                            this%lb_stripped(4):this%ub_stripped(4), &
                            this%lb_stripped(5):this%ub_stripped(5))
    end function

    function get_data_array_pointer_4d(this) result(ptr)
        !! Return a pointer of the data array class object containing 4d array
        class(data_storage_4d_t), target, intent(inout) :: this
        !! Instance of the type
        class(data_array_4d_t), pointer :: ptr

        ptr => this%data_array
    end function

    function get_pointer_4d(this) result(ptr)
        !! Returns a pointer of rank 4 pointing to the data stored in
        !! data_storage_4d_t
        class(data_storage_4d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: ptr

        ptr => this%storage
    end function

    function get_pointer_stripped_4d(this) result(ptr)
        !! Returns a pointer of rank 4 pointing to the data stored in
        !! data_storage_4d_cpu_t without ghost cells
        !! NOTE: The pointer is not contiguous anymore
        class(data_storage_4d_t), target, intent(inout) :: this
        real(kind=GP), pointer, dimension(:,:,:,:) :: ptr

        ptr(this%lb_stripped(1):, this%lb_stripped(2):, this%lb_stripped(3):, &
            this%lb_stripped(4):) &
            => this%storage(this%lb_stripped(1):this%ub_stripped(1), &
                            this%lb_stripped(2):this%ub_stripped(2), &
                            this%lb_stripped(3):this%ub_stripped(3), &
                            this%lb_stripped(4):this%ub_stripped(4))
    end function

    function get_data_array_pointer_2d(this) result(ptr)
        !! Return a pointer of the data array class object containing 2d array
        class(data_storage_2d_t), target, intent(inout) :: this
        !! Instance of the type
        class(data_array_2d_t), pointer :: ptr

        ptr => this%data_array
    end function

    function get_pointer_2d(this) result(ptr)
        !! Returns a pointer of rank 5 pointing to the data stored in
        !! data_storage_2d_t
        class(data_storage_2d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%storage
    end function

    function get_pointer_stripped_2d(this) result(ptr)
        !! Returns a pointer of rank 2 pointing to the data stored in
        !! data_storage_2d_cpu_t without ghost cells
        !! NOTE: The pointer is not contiguous anymore
        class(data_storage_2d_t), target, intent(inout) :: this
        real(kind=GP), pointer, dimension(:,:) :: ptr

        ptr(this%lb_stripped(1):, this%lb_stripped(2):) &
            => this%storage(this%lb_stripped(1):this%ub_stripped(1), &
                            this%lb_stripped(2):this%ub_stripped(2))
    end function

    function lbound_stripped_5d(this) result(lb_stripped)
        !! Returns the lower bound of the data storage without ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: lb_stripped(5)

        lb_stripped(1:5) = this%lb_stripped(1:5)
    end function

    function ubound_stripped_5d(this) result(ub_stripped)
        !! Returns the upper bound of the data storage without ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: ub_stripped(5)

        ub_stripped(1:5) = this%ub_stripped(1:5)
    end function

    function lbound_5d(this) result(lb)
        !! Returns the lower bound of the data storage with ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: lb(5)

        lb(1:5) = this%lb(1:5)
    end function

    function ubound_5d(this) result(ub)
        !! Returns the upper bound of the data storage with ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: ub(5)

        ub(1:5) = this%ub(1:5)
    end function

    function shape_5d(this) result(shp)
        !! Returns the upper bound of the data storage with ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: shp(5)

        shp = shape(this%storage)
    end function

    function shape_stripped_5d(this) result(shp)
        !! Returns the shape of the data storage without ghost cells
        class(data_storage_5d_t) :: this
        !! Instance of the type
        integer :: shp(5)

        shp = this%ub_stripped - this%lb_stripped + 1
    end function

    function lbound_stripped_4d(this) result(lb_stripped)
        !! Returns the lower bound of the data storage without ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: lb_stripped(4)

        lb_stripped(1:4) = this%lb_stripped(1:4)
    end function

    function ubound_stripped_4d(this) result(ub_stripped)
        !! Returns the upper bound of the data storage without ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: ub_stripped(4)

        ub_stripped(1:4) = this%ub_stripped(1:4)
    end function

    function lbound_4d(this) result(lb)
        !! Returns the lower bound of the data storage with ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: lb(4)

        lb(1:4) = this%lb(1:4)
    end function

    function ubound_4d(this) result(ub)
        !! Returns the upper bound of the data storage with ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: ub(4)

        ub(1:4) = this%ub(1:4)
    end function

    function shape_4d(this) result(shp)
        !! Returns the upper bound of the data storage with ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: shp(4)

        shp = shape(this%storage)
    end function

    function shape_stripped_4d(this) result(shp)
        !! Returns the shape of the data storage without ghost cells
        class(data_storage_4d_t) :: this
        !! Instance of the type
        integer :: shp(4)

        shp = this%ub_stripped - this%lb_stripped + 1
    end function

    function lbound_stripped_2d(this) result(lb_stripped)
        !! Returns the lower bound of the data storage without ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: lb_stripped(2)

        lb_stripped(1:2) = this%lb_stripped(1:2)
    end function

    function ubound_stripped_2d(this) result(ub_stripped)
        !! Returns the upper bound of the data storage without ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: ub_stripped(2)

        ub_stripped(1:2) = this%ub_stripped(1:2)
    end function

    function lbound_2d(this) result(lb)
        !! Returns the lower bound of the data storage with ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: lb(2)

        lb(1:2) = this%lb(1:2)
    end function

    function ubound_2d(this) result(ub)
        !! Returns the upper bound of the data storage with ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: ub(2)

        ub(1:2) = this%ub(1:2)
    end function

    function shape_2d(this) result(shp)
        !! Returns the shape of the data storage with ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: shp(2)

        shp = shape(this%storage)
    end function

    function shape_stripped_2d(this) result(shp)
        !! Returns the shape of the data storage without ghost cells
        class(data_storage_2d_t) :: this
        !! Instance of the type
        integer :: shp(2)

        shp = this%ub_stripped - this%lb_stripped + 1
    end function

    pure function get_size(this) result(res)
      !! Returns the number of dimensions
      class(data_storage_t), intent(in) :: this
      !! Instance of the type
      integer :: res

      res = this%size
    end function

    pure function get_dimension(this) result(res)
        !! Returns the total number of elements in storage
        class(data_storage_t), intent(in) :: this
        !! Instance of the type
        integer :: res

        res = this%dimensions
    end function

    function get_num_mail_partners(this) result(res)
        !! Returns the number of ghost cells
        class(data_storage_t) :: this
        !! Instance of the type
        integer, dimension(5) :: res

        res(1:this%dimensions) = this%number_of_mail_partners
    end function

#ifdef ENABLE_GPU
    function get_data_storage_5d_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the data_storage_gpu_5d_t
        !! C++ class instance pointer
        class(data_storage_gpu_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%data_storage_cxx_pptr
    end function

    function get_data_storage_4d_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the data_storage_gpu_4d_t
        !! C++ class instance pointer
        class(data_storage_gpu_4d_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%data_storage_cxx_pptr
    end function

    function get_data_storage_2d_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the data_storage_gpu_2d_t
        !! C++ class instance pointer
        class(data_storage_gpu_2d_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%data_storage_cxx_pptr
    end function
#endif

    subroutine update_host_5d(this)
        !! Update the 5D array on CPU from GPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_5d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_host()
    end subroutine

    subroutine update_device_5d(this)
        !! Update the 5D array on GPU from CPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_5d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_device()
    end subroutine

    subroutine update_host_4d(this)
        !! Update the 4D array on CPU from GPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_4d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_host()
    end subroutine

    subroutine update_device_4d(this)
        !! Update the 4D array on GPU from CPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_4d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_device()
    end subroutine

    subroutine update_host_2d(this)
        !! Update the 2D array on CPU from GPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_2d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_host()
    end subroutine

    subroutine update_device_2d(this)
        !! Update the 5D array on GPU from CPU. This does nothing if
        !! no specific GPU backend is set.
        class(data_storage_2d_t), intent(inout) :: this
        !! Instance of the type

        call this%data_array%update_device()
    end subroutine

end module data_storage_m
