submodule (data_storage_m) data_storage_gpu_2d_s
    use, intrinsic :: iso_c_binding, only: c_loc
    use profiler_m, only: profiler_inject
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_START_EXCH, GPU_ERR_FINISH_EXCH, &
                                      GPU_ERR_FINALIZE

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_data_storage_2d_initialize( &
            dcomm_handler_cxx_pptr, ds_data, da_cxx_pptr, ds_cxx_pptr) &
            bind(C, name="cbind_data_storage_2d_initialize")
            !! Fortran/C++ interoperable routine for the initialization
            !! of data_storage_gpu_2d_t C++ class
            import :: C_INT32_T, C_PTR, data_storage_data_t
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            !! C pointer to the dcomm_handler_t C++ class instance pointer
            type(data_storage_data_t), intent(in) :: ds_data
            !! Object of Fortran/C++ interoperable
            !! structure based on data_storage_data_t
            type(C_PTR), intent(inout) :: da_cxx_pptr
            !! C pointer to the data_array_t C++ class instance pointer
            type(C_PTR), intent(inout) :: ds_cxx_pptr
            !! C pointer to the data_storage_t C++ class instance pointer
        end function

        integer(kind=C_INT32_T) function cbind_data_storage_2d_start_exchange( &
            ds_cxx_pptr) bind(C, name="cbind_data_storage_2d_start_exchange")
            !! Fortran/C++ interoperable routine for the start_exchange routine
            !! of data_storage_gpu_2d_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: ds_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_data_storage_2d_finish_exchange(&
            ds_cxx_pptr) bind(C, name="cbind_data_storage_2d_finish_exchange")
            !! Fortran/C++ interoperable routine for the finish_exchange routine
            !! of data_storage_gpu_2d_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: ds_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_data_storage_2d_finalize( &
            ds_cxx_pptr) bind(C, name="cbind_data_storage_finalize")
            !! Fortran/C++ interoperable routine for the finalization of
            !! the data_storage_gpu_2d_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: ds_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_gpu_2d(this, dcomm_handler, init_value)
        class(data_storage_gpu_2d_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        real(kind=GP), optional, intent(in) :: init_value

        integer :: ierr
        type(data_storage_data_t), allocatable :: ds_data

        call this%initialize_parent(dcomm_handler, dimensions=2)

        allocate(this%data_array)
        call this%data_array%initialize(this%lb, this%ub, &
                                        this%lb_stripped, this%ub_stripped, &
                                        val=init_value)
        this%storage => this%data_array%get_pointer()

        ! Expose class members to the coressponding interoperable structure
        allocate(ds_data)
        call expose_data(this, ds_data)

        ! Initialize data_storage_gpu_2d_t C++ class,
        ! including deep copy to the device
        ierr = cbind_data_storage_2d_initialize( &
                   this%dcomm_handler%get_cxx_pointer(), &
                   ds_data, &
                   this%data_array%get_cxx_pointer(), &
                   this%data_storage_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

        deallocate(ds_data)

    contains

        subroutine expose_data(this, ds_data)
            !! Expose class members of data_storage_t to a Fortran/C++
            !! interoperable structure based on data_storage_data_t
            class(data_storage_gpu_2d_t), target, intent(inout) :: this
            !! Instance of the type
            type(data_storage_data_t), intent(inout) :: ds_data
            !! Fortran/C++ interoperable structure of data_storage_t

            ds_data%array_dim = this%dimensions
            ds_data%n_ex_dims = 1
            ds_data%number_of_elements_ptr = c_loc(this%number_of_elements)
            ds_data%number_of_ghost_cells_ptr = &
                c_loc(this%number_of_ghost_cells)
            ds_data%number_of_data_cells_ptr = c_loc(this%number_of_data_cells)
            ds_data%number_of_mail_partners_ptr = &
                c_loc(this%number_of_mail_partners)
            ds_data%dim_permut_ptr = c_loc(this%dim_permut)
        end subroutine

    end subroutine

    module subroutine start_exchange_gpu_2d(this)
        class(data_storage_gpu_2d_t), intent(inout) :: this
        integer :: ierr

        ierr = cbind_data_storage_2d_start_exchange(this%data_storage_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_START_EXCH, __LINE__, __FILE__)
        endif

        call profiler_inject("pack_2d", ierr)
        call profiler_inject("send_2d", ierr)

    end subroutine

    module subroutine finish_exchange_gpu_2d(this)
        class(data_storage_gpu_2d_t), intent(inout) :: this
        integer :: ierr

        ierr = cbind_data_storage_2d_finish_exchange(this%data_storage_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINISH_EXCH, __LINE__, __FILE__)
        endif

        call profiler_inject("receive_2d", ierr)
        call profiler_inject("unpack_2d", ierr)

    end subroutine

    module subroutine finalize_gpu_2d(this)
        type(data_storage_gpu_2d_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator C++ class
        ierr = cbind_data_storage_2d_finalize(this%data_storage_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
