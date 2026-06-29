module data_array_m
    !! Module containing the definition and interface of the data array types.
    !! Data array types are wrapper classes containing the 2d, 4d, 5d array and
    !! their bounds.
    use, intrinsic :: iso_fortran_env, only: INT64
    use, intrinsic :: iso_c_binding, only: C_PTR, c_loc, c_f_pointer
    use genex_fortran_env_m, only: GP
    use params_gpu_offload_m, only : get_use_gpu_offload
#ifdef ENABLE_GPU
    use data_array_gpu_m, only: cbind_data_array_get_device_pointer
#endif

    implicit none
    private

    type, public, abstract :: data_array_t
        !! Type to store the data of a multidimensional (2d, 4d, or 5d) array
        integer, private :: array_dim
        !! Number of dimension or rank of the array
        integer(kind=INT64), private :: array_size
        !! Total number of elements of the array
        integer(kind=INT64), private :: array_size_stripped
        !! Total number of elements of the array without ghost
        integer, private, dimension(:), allocatable :: array_shape
        !! Array specifying the number of elements for each dimension
        integer, private, dimension(:), allocatable :: array_shape_stripped
        !! Array specifying the number of elements without ghost
        !! for each dimension
        integer, private, dimension(:), allocatable :: array_lb
        !! Array specifying the lower boundary index for each dimension
        integer, private, dimension(:), allocatable :: array_ub
        !! Array specifying the upper boundary index for each dimension
        integer, private, dimension(:), allocatable :: array_lb_stripped
        !! Array specifying the lower boundary index without ghost for
        !! each dimension
        integer, private, dimension(:), allocatable :: array_ub_stripped
        !! Array specifying the upper boundary index without ghost for
        !! each dimension
        logical, private :: is_distributed_array = .false.
        !! Bookeeping variable to indicate if the array is distributed array
#ifdef ENABLE_GPU
        type(C_PTR), private :: data_array_cxx_pptr
        !! C pointer to the data_array_t C++ class instance pointer
#endif

    contains

        procedure, public :: get_dimension
        procedure, public :: get_size
        procedure, public :: get_size_stripped
        procedure, public :: get_shape
        procedure, public :: get_shape_stripped
        procedure, public :: get_lbound
        procedure, public :: get_ubound
        procedure, public :: get_lbound_stripped
        procedure, public :: get_ubound_stripped
        procedure, public :: is_distributed
        procedure, private :: initialize_base
        procedure, private :: initialize_local
        procedure, private :: initialize_distributed
        procedure, private :: initialize_with_mold
        generic, public :: initialize => initialize_local, &
                                         initialize_distributed, &
                                         initialize_with_mold
        procedure, public :: update_host
        procedure, public :: update_device
        procedure, private :: finalize

#ifdef ENABLE_GPU
        procedure, private :: initialize_gpu
        procedure, public :: get_cxx_pointer => get_data_array_cxx_pointer
        procedure, public :: get_readonly_cxx_pointer => &
                             get_data_array_readonly_cxx_pointer
#endif
    end type

    interface
        module subroutine initialize_local(this, lb, ub, val)
            !! Initialize the type
            class(data_array_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(:), intent(in) :: lb
            !! Lower boundary index array
            integer, dimension(:), intent(in) :: ub
            !! Upper boundary index array
            real(kind=GP), optional, intent(in) :: val
            !! Initial uniform value of the array
        end subroutine

        module subroutine initialize_distributed(this, lb, ub, &
                                                 lb_stripped, ub_stripped, val)
            !! Initialize the type
            class(data_array_t), intent(inout) :: this
            integer, dimension(:), intent(in) :: lb
            integer, dimension(:), intent(in) :: ub
            integer, dimension(:), intent(in) :: lb_stripped
            !! Lower boundary index array without ghost
            integer, dimension(:), intent(in) :: ub_stripped
            !! Upper boundary index array without ghost
            real(kind=GP), optional, intent(in) :: val
        end subroutine

        module subroutine initialize_with_mold(this, mold, val)
            !! Initialize the type
            class(data_array_t), intent(inout) :: this
            class(data_array_t), intent(in) :: mold
            !! Instance of the type which the array properties are based from
            real(kind=GP), optional, intent(in) :: val
        end subroutine

        module subroutine initialize_base(this, lb, ub, val)
            !! Initialize the type
            class(data_array_t), intent(inout) :: this
            integer, dimension(:), intent(in) :: lb
            integer, dimension(:), intent(in) :: ub
            real(kind=GP), optional, intent(in) :: val
        end subroutine

        module subroutine update_host(this)
            !! Update the array on CPU from GPU. This does nothing if
            !! no specific GPU backend is set.
            class(data_array_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device(this)
            !! Update the array on GPU from CPU. This does nothing if
            !! no specific GPU backend is set.
            class(data_array_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine finalize(this)
            !! Finalize the data_array_t class, including C++ deallocation from
            !! the device memory if applicable
            class(data_array_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

#ifdef ENABLE_GPU
        module subroutine initialize_gpu(this, val)
            !! Initialize the data_array_t C++ class, including allocation on
            !! the device memory
            class(data_array_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), optional, intent(in) :: val
        end subroutine
#endif

    end interface

    type, public, extends(data_array_t) :: data_array_2d_t
        !! Type to store the data of a 2 dimensional array
        real(kind=GP), private, dimension(:,:), pointer :: array
        !! 2 dimensional array stored in the data array type
    contains
        procedure, public :: get_pointer => get_pointer_2d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_2d
        procedure, public :: get_readonly_pointer => get_readonly_pointer_2d
        procedure, public :: get_readonly_pointer_stripped => &
                             get_readonly_pointer_stripped_2d
        procedure, public :: get_device_pointer => get_device_pointer_2d
        procedure, public :: get_readonly_device_pointer => &
                             get_readonly_device_pointer_2d
        final :: finalize_2d
    end type

    interface
        module subroutine finalize_2d(this)
            type(data_array_2d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(data_array_t) :: data_array_3d_t
        !! Type to store the data of a 3 dimensional array
        real(kind=GP), private, dimension(:,:,:), pointer :: array
        !! 3 dimensional array stored in the data array type
    contains
        procedure, public :: get_pointer => get_pointer_3d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_3d
        procedure, public :: get_readonly_pointer => get_readonly_pointer_3d
        procedure, public :: get_readonly_pointer_stripped => &
                             get_readonly_pointer_stripped_3d
        procedure, public :: get_device_pointer => get_device_pointer_3d
        procedure, public :: get_readonly_device_pointer => &
                             get_readonly_device_pointer_3d
        final :: finalize_3d
    end type

    interface
        module subroutine finalize_3d(this)
            type(data_array_3d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(data_array_t) :: data_array_4d_t
        !! Type to store the data of a 4 dimensional array
        real(kind=GP), private, dimension(:,:,:,:), pointer :: array
        !! 4 dimensional array stored in the data array type
    contains
        procedure, public :: get_pointer => get_pointer_4d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_4d
        procedure, public :: get_readonly_pointer => get_readonly_pointer_4d
        procedure, public :: get_readonly_pointer_stripped => &
                             get_readonly_pointer_stripped_4d
        procedure, public :: get_device_pointer => get_device_pointer_4d
        procedure, public :: get_readonly_device_pointer => &
                             get_readonly_device_pointer_4d
        final :: finalize_4d
    end type

    interface
        module subroutine finalize_4d(this)
            type(data_array_4d_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(data_array_t) :: data_array_5d_t
        !! Type to store the data of a 5 dimensional array
        real(kind=GP), private, dimension(:,:,:,:,:), pointer :: array
        !! 5 dimensional array stored in the data array type
    contains
        procedure, public :: get_pointer => get_pointer_5d
        procedure, public :: get_pointer_stripped => get_pointer_stripped_5d
        procedure, public :: get_readonly_pointer => get_readonly_pointer_5d
        procedure, public :: get_readonly_pointer_stripped => &
                             get_readonly_pointer_stripped_5d
        procedure, public :: get_device_pointer => get_device_pointer_5d
        procedure, public :: get_readonly_device_pointer => &
                             get_readonly_device_pointer_5d
        final :: finalize_5d
    end type

    interface
        module subroutine finalize_5d(this)
            type(data_array_5d_t), intent(inout) :: this
        end subroutine
    end interface

contains

    pure integer function get_dimension(this)
        !! Returns the number of dimension or rank of the array
        class(data_array_t), intent(in) :: this
        !! Instance of the type

        get_dimension = this%array_dim
    end function

    pure integer(kind=INT64) function get_size(this)
        !! Returns the total number of elements of the array
        class(data_array_t), intent(in) :: this
        !! Instance of the type

        get_size = this%array_size
    end function

    pure integer(kind=INT64) function get_size_stripped(this)
        !! Returns the total number of elements of the array without ghost
        class(data_array_t), intent(in) :: this
        !! Instance of the type

        get_size_stripped = this%array_size_stripped
    end function

    pure function get_shape(this) result(arr_shp)
        !! Returns the array specifying the number of elements for
        !! each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: arr_shp(this%array_dim)

        arr_shp(1:this%array_dim) = this%array_shape(1:this%array_dim)
    end function

    pure function get_shape_stripped(this) result(arr_shp_stripped)
        !! Returns the array specifying the number of elements without ghost
        !! for each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: arr_shp_stripped(this%array_dim)

        arr_shp_stripped(1:this%array_dim) = &
            this%array_shape_stripped(1:this%array_dim)
    end function

    pure function get_lbound(this) result(lb)
        !! Returns the array specifying the lower boundary index for
        !! each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: lb(this%array_dim)

        lb(1:this%array_dim) = this%array_lb(1:this%array_dim)
    end function

    pure function get_ubound(this) result(ub)
        !! Returns the array specifying the upper boundary index for
        !! each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: ub(this%array_dim)

        ub(1:this%array_dim) = this%array_ub(1:this%array_dim)
    end function

    pure function get_lbound_stripped(this) result(lb_stripped)
        !! Returns the array specifying the lower boundary index
        !! without ghost for each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: lb_stripped(this%array_dim)

        lb_stripped(1:this%array_dim) = this%array_lb_stripped(1:this%array_dim)
    end function

    pure function get_ubound_stripped(this) result(ub_stripped)
        !! Returns the array specifying the upper boundary index
        !! without ghost for each dimension
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        integer :: ub_stripped(this%array_dim)

        ub_stripped(1:this%array_dim) = this%array_ub_stripped(1:this%array_dim)
    end function

    pure function is_distributed(this) result(res)
        !! Returns true if the array is distributed array, otherwise false
        class(data_array_t), intent(in) :: this
        !! Instance of the type
        logical :: res

        res = this%is_distributed_array
    end function

    function get_pointer_2d(this) result(ptr)
        !! Returns the pointer to the 2d array
        class(data_array_2d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%array
    end function

    function get_pointer_stripped_2d(this) result(ptr)
        !! Returns the pointer to the 2d array without the ghost cells
        class(data_array_2d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:) :: ptr

        ptr(this%array_lb_stripped(1):,this%array_lb_stripped(2):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2))
    end function

    function get_readonly_pointer_2d(this) result(ptr)
        !! Returns the readonly pointer to the 2d array
        class(data_array_2d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%array
    end function

    function get_readonly_pointer_stripped_2d(this) result(ptr)
        !! Returns the readonly pointer to the 2d array without the ghost cells
        class(data_array_2d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:) :: ptr

        ptr(this%array_lb_stripped(1):,this%array_lb_stripped(2):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2))
    end function

    function get_device_pointer_2d(this) result(dev_ptr)
        !! Returns the device pointer of the 2d array
        class(data_array_2d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):) => dev_ptr
        endif
#endif
    end function

    function get_readonly_device_pointer_2d(this) result(dev_ptr)
        !! Returns the readonly device pointer of the 2d array
        class(data_array_2d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):) => dev_ptr
        endif
#endif
    end function

    function get_pointer_3d(this) result(ptr)
        !! Returns the pointer to the 3d array
        class(data_array_3d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: ptr

        ptr => this%array
    end function

    function get_pointer_stripped_3d(this) result(ptr)
        !! Returns the pointer to the 3d array without the ghost cells
        class(data_array_3d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3))
    end function

    function get_readonly_pointer_3d(this) result(ptr)
        !! Returns the readonly pointer to the 3d array
        class(data_array_3d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: ptr

        ptr => this%array
    end function

    function get_readonly_pointer_stripped_3d(this) result(ptr)
        !! Returns the readonly pointer to the 3d array without the ghost cells
        class(data_array_3d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3))
    end function

    function get_device_pointer_3d(this) result(dev_ptr)
        !! Returns the device pointer of the 3d array
        class(data_array_3d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):) => dev_ptr
        endif
#endif
    end function

    function get_readonly_device_pointer_3d(this) result(dev_ptr)
        !! Returns the readonly device pointer of the 3d array
        class(data_array_3d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):) => dev_ptr
        endif
#endif
    end function

    function get_pointer_4d(this) result(ptr)
        !! Returns the pointer to the 4d array
        class(data_array_4d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: ptr

        ptr => this%array
    end function

    function get_pointer_stripped_4d(this) result(ptr)
        !! Returns the pointer to the 4d array without the ghost cells
        class(data_array_4d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):, &
            this%array_lb_stripped(4):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3), &
                          this%array_lb_stripped(4):this%array_ub_stripped(4))
    end function

    function get_readonly_pointer_4d(this) result(ptr)
        !! Returns the readonly pointer to the 4d array
        class(data_array_4d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: ptr

        ptr => this%array
    end function

    function get_readonly_pointer_stripped_4d(this) result(ptr)
        !! Returns the readonly pointer to the 4d array without the ghost cells
        class(data_array_4d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):, &
            this%array_lb_stripped(4):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3), &
                          this%array_lb_stripped(4):this%array_ub_stripped(4))
    end function

    function get_device_pointer_4d(this) result(dev_ptr)
        !! Returns the device pointer of the 4d array
        class(data_array_4d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):, this%array_lb(4):) => dev_ptr
        endif
#endif
    end function

    function get_readonly_device_pointer_4d(this) result(dev_ptr)
        !! Returns the readonly device pointer of the 4d array
        class(data_array_4d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):, this%array_lb(4):) => dev_ptr
        endif
#endif
    end function

    function get_pointer_5d(this) result(ptr)
        !! Returns the pointer to the 5d array
        class(data_array_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: ptr

        ptr => this%array
    end function

    function get_pointer_stripped_5d(this) result(ptr)
        !! Returns the pointer to the 5d array without the ghost cells
        class(data_array_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):, &
            this%array_lb_stripped(4):, &
            this%array_lb_stripped(5):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3), &
                          this%array_lb_stripped(4):this%array_ub_stripped(4), &
                          this%array_lb_stripped(5):this%array_ub_stripped(5))
    end function

    function get_readonly_pointer_5d(this) result(ptr)
        !! Returns the readonly pointer to the 5d array
        class(data_array_5d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: ptr

        ptr => this%array
    end function

    function get_readonly_pointer_stripped_5d(this) result(ptr)
        !! Returns the readonly pointer to the 5d array without the ghost cells
        class(data_array_5d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: ptr

        ptr(this%array_lb_stripped(1):, &
            this%array_lb_stripped(2):, &
            this%array_lb_stripped(3):, &
            this%array_lb_stripped(4):, &
            this%array_lb_stripped(5):) &
            => this%array(this%array_lb_stripped(1):this%array_ub_stripped(1), &
                          this%array_lb_stripped(2):this%array_ub_stripped(2), &
                          this%array_lb_stripped(3):this%array_ub_stripped(3), &
                          this%array_lb_stripped(4):this%array_ub_stripped(4), &
                          this%array_lb_stripped(5):this%array_ub_stripped(5))
    end function

    function get_device_pointer_5d(this) result(dev_ptr)
        !! Returns the device pointer of the 5d array
        class(data_array_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):, this%array_lb(4):, &
                    this%array_lb(5):) => dev_ptr
        endif
#endif
    end function

    function get_readonly_device_pointer_5d(this) result(dev_ptr)
        !! Returns the readonly device pointer of the 5d array
        class(data_array_5d_t), target, intent(in) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: dev_ptr
        type(C_PTR) :: dev_cxx_ptr, da_cxx_pptr

        dev_ptr => this%array

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            da_cxx_pptr = this%data_array_cxx_pptr
            dev_cxx_ptr = cbind_data_array_get_device_pointer(da_cxx_pptr)
            call c_f_pointer(dev_cxx_ptr, dev_ptr, this%array_shape)
            dev_ptr(this%array_lb(1):, this%array_lb(2):, &
                    this%array_lb(3):, this%array_lb(4):, &
                    this%array_lb(5):) => dev_ptr
        endif
#endif
    end function

#ifdef ENABLE_GPU
    function get_data_array_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the data_array_t
        !! C++ class instance pointer
        class(data_array_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%data_array_cxx_pptr
    end function

    function get_data_array_readonly_cxx_pointer(this) result(pptr)
        !! Returns the readonly C pointer of the data_array_t
        !! C++ class instance pointer
        class(data_array_t), target, intent(in) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%data_array_cxx_pptr
    end function
#endif

end module data_array_m
