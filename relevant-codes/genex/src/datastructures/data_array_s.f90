submodule (data_array_m) data_array_s
    use genex_status_codes_m, only: GENEX_ERR_DATA_STRUCTURE
    use genex_error_handling_m, only: handle_error, handle_error_gpu, &
                                      GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE, GPU_ERR_COPY, &
                                      error_info_t
    use type_converters_m, only: string
    use profiler_m, only: profiler_start, profiler_stop
    use op_set_uniform_m, only: op_set_uniform_cpu_t
#ifdef ENABLE_GPU
    use data_array_gpu_m, only: data_array_data_t, &
                                cbind_data_array_initialize, &
                                cbind_data_array_finalize, &
                                cbind_data_array_update_host, &
                                cbind_data_array_update_device
#endif

    implicit none

contains

    subroutine check_bounds(array_1, array_2)
        ! Check if the size of two arrays is the same
        integer, dimension(:), intent(in) :: array_1, array_2

        if (size(array_1) /= size(array_2)) then
            call handle_error("The size of the boundary indices was not the &
                              &same!", GENEX_ERR_DATA_STRUCTURE, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Size of bounds was: ", &
                                                [size(array_1), size(array_2)]))
        endif
    end subroutine

    subroutine check_dim(supported_dim, array_dim)
        ! Check if the array dimension/rank requested is supported
        integer, intent(in) :: supported_dim, array_dim

        if (array_dim /= supported_dim) then
            call handle_error("The array dimension/rank is not supported!", &
                              GENEX_ERR_DATA_STRUCTURE, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Array/supported dim: ", &
                                                [array_dim, supported_dim]))
        endif
    end subroutine

    subroutine check_type(this, mold)
        ! Check if the type of two objects is the same
        class(data_array_t), intent(in) :: this, mold

        if (.not. same_type_as(this, mold)) then
            call handle_error("Failed to initialize a data array with mold &
                              &because the mold type was not the same!", &
                              GENEX_ERR_DATA_STRUCTURE, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine initialize_local(this, lb, ub, val)
        class(data_array_t), intent(inout) :: this
        integer, dimension(:), intent(in) :: lb
        integer, dimension(:), intent(in) :: ub
        real(kind=GP), optional, intent(in) :: val

        call check_bounds(lb, ub)
        call this%initialize_base(lb, ub, val)

        this%is_distributed_array = .false.
        this%array_lb_stripped(1:this%array_dim) = lb(1:this%array_dim)
        this%array_ub_stripped(1:this%array_dim) = ub(1:this%array_dim)
        this%array_shape_stripped(1:this%array_dim) = &
            this%array_shape(1:this%array_dim)
        this%array_size_stripped = this%array_size

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Initialize the data_array_t C++ class
            call this%initialize_gpu(val)
        endif
#endif
    end subroutine

    module subroutine initialize_distributed(this, lb, ub, lb_stripped, &
                                             ub_stripped, val)
        class(data_array_t), intent(inout) :: this
        integer, dimension(:), intent(in) :: lb
        integer, dimension(:), intent(in) :: ub
        integer, dimension(:), intent(in) :: lb_stripped
        integer, dimension(:), intent(in) :: ub_stripped
        real(kind=GP), optional, intent(in) :: val

        call check_bounds(lb, ub)
        call check_bounds(lb, lb_stripped)
        call check_bounds(lb, ub_stripped)
        call this%initialize_base(lb, ub, val)

        this%is_distributed_array = .true.
        this%array_lb_stripped(1:this%array_dim) = &
            lb_stripped(1:this%array_dim)
        this%array_ub_stripped(1:this%array_dim) = &
            ub_stripped(1:this%array_dim)
        this%array_shape_stripped = (this%array_ub_stripped - &
                                     this%array_lb_stripped) + 1
        this%array_size_stripped = product(this%array_shape_stripped)

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Initialize the data_array_t C++ class
            call this%initialize_gpu(val)
        endif
#endif
    end subroutine

    module subroutine initialize_with_mold(this, mold, val)
        class(data_array_t), intent(inout) :: this
        class(data_array_t), intent(in) :: mold
        real(kind=GP), optional, intent(in) :: val

        call check_type(this, mold)
        call this%initialize_base(mold%get_lbound(), mold%get_ubound(), val)

        this%is_distributed_array = mold%is_distributed()
        this%array_size_stripped  = mold%get_size_stripped()
        this%array_lb_stripped(1:this%array_dim)    = mold%get_lbound_stripped()
        this%array_ub_stripped(1:this%array_dim)    = mold%get_ubound_stripped()
        this%array_shape_stripped(1:this%array_dim) = mold%get_shape_stripped()

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Initialize the data_array_t C++ class
            call this%initialize_gpu(val)
        endif
#endif
    end subroutine

    module subroutine initialize_base(this, lb, ub, val)
        class(data_array_t), intent(inout) :: this
        integer, dimension(:), intent(in) :: lb
        integer, dimension(:), intent(in) :: ub
        real(kind=GP), optional, intent(in) :: val

        type(op_set_uniform_cpu_t) :: op_set_uniform
        real(kind=GP) :: initial_value

        if(present(val)) then
            initial_value = val
        else
            initial_value = 0.0_GP
        endif

        this%array_dim = size(lb)
        allocate(this%array_shape(this%array_dim))
        allocate(this%array_shape_stripped(this%array_dim))
        allocate(this%array_lb(this%array_dim))
        allocate(this%array_ub(this%array_dim))
        allocate(this%array_lb_stripped(this%array_dim))
        allocate(this%array_ub_stripped(this%array_dim))

        this%array_lb(1:this%array_dim) = lb(1:this%array_dim)
        this%array_ub(1:this%array_dim) = ub(1:this%array_dim)
        this%array_shape = (this%array_ub - this%array_lb) + 1
        this%array_size  = product(this%array_shape)

        if(get_use_gpu_offload()) return

        ! Allocate the array on fortran layer in pure Fortran run
        select type(this)
            type is(data_array_2d_t)
                call check_dim(2, this%array_dim)
                allocate(this%array(this%array_lb(1):this%array_ub(1), &
                                    this%array_lb(2):this%array_ub(2)))
                call op_set_uniform%apply(this%array, initial_value)
            type is(data_array_3d_t)
                call check_dim(3, this%array_dim)
                allocate(this%array(this%array_lb(1):this%array_ub(1), &
                                    this%array_lb(2):this%array_ub(2), &
                                    this%array_lb(3):this%array_ub(3)))
                call op_set_uniform%apply(this%array, initial_value)
            type is(data_array_4d_t)
                call check_dim(4, this%array_dim)
                allocate(this%array(this%array_lb(1):this%array_ub(1), &
                                    this%array_lb(2):this%array_ub(2), &
                                    this%array_lb(3):this%array_ub(3), &
                                    this%array_lb(4):this%array_ub(4)))
                call op_set_uniform%apply(this%array, initial_value)
            type is(data_array_5d_t)
                call check_dim(5, this%array_dim)
                allocate(this%array(this%array_lb(1):this%array_ub(1), &
                                    this%array_lb(2):this%array_ub(2), &
                                    this%array_lb(3):this%array_ub(3), &
                                    this%array_lb(4):this%array_ub(4), &
                                    this%array_lb(5):this%array_ub(5)))
                call op_set_uniform%apply(this%array, initial_value)
        end select
    end subroutine

    module subroutine finalize(this)
        class(data_array_t), intent(inout) :: this

        integer :: ierr

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ierr = cbind_data_array_finalize(this%data_array_cxx_pptr)

            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
            endif
        endif
#endif

        if(get_use_gpu_offload()) return

        ! Deallocate the array which were allocated on Fortran layer
        select type(this)
            type is(data_array_2d_t)
                deallocate(this%array)
            type is(data_array_3d_t)
                deallocate(this%array)
            type is(data_array_4d_t)
                deallocate(this%array)
            type is(data_array_5d_t)
                deallocate(this%array)
        end select

    end subroutine

    module subroutine finalize_2d(this)
        type(data_array_2d_t), intent(inout) :: this
        call this%finalize()
    end subroutine

    module subroutine finalize_3d(this)
        type(data_array_3d_t), intent(inout) :: this
        call this%finalize()
    end subroutine

    module subroutine finalize_4d(this)
        type(data_array_4d_t), intent(inout) :: this
        call this%finalize()
    end subroutine

    module subroutine finalize_5d(this)
        type(data_array_5d_t), intent(inout) :: this
        call this%finalize()
    end subroutine

    module subroutine update_host(this)
        class(data_array_t), intent(inout) :: this

        character(len=:), allocatable :: region_name
        integer :: ierr_c, ierr_profiler

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            region_name = "update_host_"//string(this%array_dim)//"d"
            call profiler_start(region_name, ierr_profiler)
            ierr_c = cbind_data_array_update_host(this%data_array_cxx_pptr)
            call profiler_stop(region_name, ierr_profiler)

            if(ierr_c /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
        endif
#endif
    end subroutine

    module subroutine update_device(this)
        class(data_array_t), intent(inout) :: this

        character(len=:), allocatable :: region_name
        integer :: ierr_c, ierr_profiler

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            region_name = "update_device_"//string(this%array_dim)//"d"
            call profiler_start(region_name, ierr_profiler)
            ierr_c = cbind_data_array_update_device(this%data_array_cxx_pptr)
            call profiler_stop(region_name, ierr_profiler)

            if(ierr_c /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
        endif
#endif
    end subroutine

#ifdef ENABLE_GPU
    module subroutine initialize_gpu(this, val)
        class(data_array_t), intent(inout) :: this
        real(kind=GP), optional, intent(in) :: val

        type(data_array_data_t), allocatable :: da_data
        integer :: ierr

        ! Expose class members to the coressponding interoperable structure
        allocate(da_data)
        call expose_data(this, val, da_data)

        ! Initialize data_array_t C++ class, including deep copy to the device
        ierr = cbind_data_array_initialize(da_data, &
                                           this%data_array_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

        ! Convert the C++ allocated array pointer to Fortran type and shift to
        ! the lower bounds
        select type(this)
            type is(data_array_2d_t)
                call check_dim(2, this%array_dim)
                call c_f_pointer(da_data%array_ptr, this%array, &
                                 this%array_shape)
                this%array(this%array_lb(1):, this%array_lb(2):) => this%array
            type is(data_array_3d_t)
                call check_dim(3, this%array_dim)
                call c_f_pointer(da_data%array_ptr, this%array, &
                                 this%array_shape)
                this%array(this%array_lb(1):, this%array_lb(2):, &
                            this%array_lb(3):) => this%array
            type is(data_array_4d_t)
                call check_dim(4, this%array_dim)
                call c_f_pointer(da_data%array_ptr, this%array, &
                                 this%array_shape)
                this%array(this%array_lb(1):, this%array_lb(2):, &
                            this%array_lb(3):, this%array_lb(4):) => this%array
            type is(data_array_5d_t)
                call check_dim(5, this%array_dim)
                call c_f_pointer(da_data%array_ptr, this%array, &
                                 this%array_shape)
                this%array(this%array_lb(1):, this%array_lb(2):, &
                            this%array_lb(3):, this%array_lb(4):, &
                            this%array_lb(5):) => this%array
        end select

        deallocate(da_data)

    contains

        subroutine expose_data(this, val, da_data)
            !! Expose class members of data_array_t to a Fortran/C++
            !! interoperable structure based on data_array_data_t
            class(data_array_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), optional, intent(in) :: val
            !! Initial uniform value of the array
            type(data_array_data_t), intent(inout) :: da_data
            !! Fortran/C++ interoperable structure of data_array_t

            da_data%array_dim            = this%array_dim
            da_data%array_size           = this%array_size
            da_data%array_size_stripped  = this%array_size_stripped
            da_data%is_distributed_array = &
                merge(1, 0, this%is_distributed_array)
            da_data%array_shape_ptr          = c_loc(this%array_shape)
            da_data%array_shape_stripped_ptr = c_loc(this%array_shape_stripped)
            da_data%array_lb_ptr             = c_loc(this%array_lb)
            da_data%array_ub_ptr             = c_loc(this%array_ub)
            da_data%array_lb_stripped_ptr    = c_loc(this%array_lb_stripped)
            da_data%array_ub_stripped_ptr    = c_loc(this%array_ub_stripped)

            da_data%init_value = 0.0_GP
            if(present(val)) then
                da_data%init_value = val
            endif
        end subroutine

    end subroutine
#endif

end submodule
