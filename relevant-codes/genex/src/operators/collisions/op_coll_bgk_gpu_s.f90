submodule(op_coll_m) op_coll_bgk_gpu_s
    !! Contains implementation of op_coll_bgk_gpu_t which represents the BGK
    !! collision operator.

    use, intrinsic :: iso_fortran_env, only: INT64
    use, intrinsic :: iso_c_binding, only: C_INT32_T
    use params_collisions_m, only: get_relx_type
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_coll_bgk_initialize( &
            dcomm_handler_cxx_pptr, mesh_cxx_pptr, da_collog_cxx_pptr, &
            op_cxx_pptr) bind(C, name="cbind_op_coll_bgk_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_coll_bgk_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(inout) :: da_collog_cxx_pptr
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_coll_bgk_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_coll_bgk_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_coll_bgk_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_coll_bgk_apply( &
            op_cxx_pptr, da_f_in_cxx_pptr, da_moments_cxx_pptr, &
            da_f_out_cxx_pptr) bind(C, name="cbind_op_coll_bgk_apply")
            !! Fortran/C++ interoperable apply routine for op_coll_bgk_gpu_t
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: op_cxx_pptr
            type(C_PTR), intent(in) :: da_f_in_cxx_pptr
            type(C_PTR), intent(in) :: da_moments_cxx_pptr
            type(C_PTR), intent(inout) :: da_f_out_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_coll_bgk_gpu(this, dcomm_handler, mesh)
        class(op_coll_bgk_gpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: ierr, lb_stripped(5), ub_stripped(5)

        this%n_flops = 23 ! We estimate exp with 10 and division with 4
        this%n_ls    = 5

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        ! Check relaxation type
        if(get_relx_type() /= "et" .and. get_relx_type() /= "em") then
            call handle_error("Selected relaxation type "//get_relx_type() &
                              //" is not supported for BGK collisions!", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        lb_stripped = this%dcomm_handler%get_lbound_stripped()
        ub_stripped = this%dcomm_handler%get_ubound_stripped()

        allocate(this%da_collog)
        call this%da_collog%initialize(lb_stripped(1:2), ub_stripped(1:2))

        ! Initialize operator member arrays on the device memory
        ierr = cbind_op_coll_bgk_initialize(dcomm_handler%get_cxx_pointer(), &
                                            mesh%get_cxx_pointer(), &
                                            this%da_collog%get_cxx_pointer(), &
                                            this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

    end subroutine

    module subroutine apply_coll_bgk_gpu(this, da_f_in, da_moments, da_f_out)
        class(op_coll_bgk_gpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: ierr

        this%n_iterations = da_f_in%get_size_stripped() &
                          * int(this%mesh%size_sp(), kind=INT64)
        call this%perf_counter%start_measurement()

        ierr = cbind_op_coll_bgk_apply(this%op_cxx_pptr, &
                                       da_f_in%get_readonly_cxx_pointer(), &
                                       da_moments%get_cxx_pointer(), &
                                       da_f_out%get_cxx_pointer())

        call this%perf_counter%end_measurement()

    end subroutine

    module subroutine finalize_coll_bgk_gpu(this)
        type(op_coll_bgk_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on the device memory
        ierr = cbind_op_coll_bgk_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
