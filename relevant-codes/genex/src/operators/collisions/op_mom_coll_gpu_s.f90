submodule(op_mom_coll_m) op_mom_coll_gpu_s
    !! Contains implementation of op_mom_coll_gpu_t
    use, intrinsic :: iso_c_binding, only: C_INT32_T
    use profiler_m, only: profiler_inject, profiler_inject_allreduce
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE
    use data_array_m, only: data_array_4d_t

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_mom_coll_initialize( &
            dcomm_handler_cxx_pptr, mesh_cxx_pptr, op_cxx_pptr) &
            bind(C, name="cbind_op_mom_coll_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_mom_coll_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_mom_coll_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_mom_coll_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_mom_coll_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_mom_coll_apply( &
            dcomm_handler_cxx_pptr, mesh_cxx_pptr, op_cxx_pptr, &
            da_f_in_cxx_pptr, da_moments_cxx_pptr) &
            bind(C, name="cbind_op_mom_coll_apply")
            !! Fortran/C++ interoperable apply routine for op_mom_coll_gpu_t
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(in) :: op_cxx_pptr
            type(C_PTR), intent(in) :: da_f_in_cxx_pptr
            type(C_PTR), intent(inout) :: da_moments_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_gpu(this, dcomm_handler, mesh)
        class(op_mom_coll_gpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: ierr

        this%n_flops = 17
        this%n_ls    = 6

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        ierr = cbind_op_mom_coll_initialize(dcomm_handler%get_cxx_pointer(), &
                                            mesh%get_cxx_pointer(), &
                                            this%op_cxx_pptr)
        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

    end subroutine

    module subroutine apply_gpu(this, da_f_in, da_moments)
        class(op_mom_coll_gpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments

        integer :: ierr, n_sp
        integer, dimension(5) :: lb_stripped, ub_stripped
        class(data_array_4d_t), allocatable :: da_send_buffer

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()
        n_sp = this%mesh%size_sp()

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        ierr = cbind_op_mom_coll_apply(this%dcomm_handler%get_cxx_pointer(), &
                                       this%mesh%get_cxx_pointer(), &
                                       this%op_cxx_pptr, &
                                       da_f_in%get_readonly_cxx_pointer(), &
                                       da_moments%get_cxx_pointer())

        call profiler_inject("reduction", ierr)
        call profiler_inject_allreduce(ierr)
        call profiler_inject("store", ierr)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_gpu(this)
        type(op_mom_coll_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on the device memory
        ierr = cbind_op_mom_coll_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
