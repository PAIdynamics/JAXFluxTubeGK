submodule (op_mom_ohms_law_m) op_mom_ohms_law_gpu_s
    use MPI
    use, intrinsic :: iso_c_binding, only: C_INT32_T
    use genex_fortran_env_m, only: CP
    use data_array_m, only: data_array_3d_t
    use params_species_m, only: get_mass, get_charge
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_beta_ref
    use profiler_m, only: profiler_inject, profiler_inject_allreduce
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_mom_ohms_law_initialize( &
            mesh_cxx_pptr, op_cxx_pptr) &
            bind(C, name="cbind_op_mom_ohms_law_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_mom_ohms_law_gpu_t C++ class
            import :: C_PTR, C_INT32_T, CP
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_mom_ohms_law_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_mom_ohms_law_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_mom_ohms_law_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_mom_ohms_law_apply( &
            mesh_cxx_pptr, dcomm_handler_cxx_pptr, op_cxx_pptr, &
            da_f_in_cxx_pptr, da_dfdt_in_cxx_pptr, &
            da_lambda_ohms_law_cxx_pptr, da_b_ohms_law_cxx_pptr) &
            bind(C, name="cbind_op_mom_ohms_law_apply")
            !! Fortran/C++ interoperable apply routine for
            !! op_mom_ohms_law_gpu_t
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            type(C_PTR), intent(in) :: op_cxx_pptr
            type(C_PTR), intent(in) :: da_f_in_cxx_pptr
            type(C_PTR), intent(in) :: da_dfdt_in_cxx_pptr
            type(C_PTR), intent(inout) :: da_lambda_ohms_law_cxx_pptr
            type(C_PTR), intent(inout) :: da_b_ohms_law_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_gpu(this, dcomm_handler, mesh, bsg_op)
        class(op_mom_ohms_law_gpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op
        integer :: n, n_points_sp, ierr

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        this%bsg_op => bsg_op

        ! Initialize operator member arrays on the device memory
        ierr = cbind_op_mom_ohms_law_initialize(this%mesh%get_cxx_pointer(),&
                                                this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

    end subroutine initialize_gpu

    module subroutine apply_gpu(this, da_f_in, da_dfdt_in, da_lambda_ohms_law, &
                                da_b_ohms_law)
        class(op_mom_ohms_law_gpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_5d_t), intent(in) :: da_dfdt_in
        class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law

        integer :: ierr
        integer, dimension(5) :: lb_stripped, ub_stripped
        class(data_array_3d_t), allocatable :: da_send_buffer, da_receive_buffer
        type(C_PTR) :: da_f_in_cxx_pptr, da_dfdt_in_cxx_pptr

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        ! NOTE: Calling these getters directly at function call below results in
        !       compile-time error with Intel release mode.
        da_f_in_cxx_pptr = da_f_in%get_readonly_cxx_pointer()
        da_dfdt_in_cxx_pptr = da_dfdt_in%get_readonly_cxx_pointer()

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        ierr = cbind_op_mom_ohms_law_apply( &
                   this%mesh%get_cxx_pointer(), &
                   this%dcomm_handler%get_cxx_pointer(), &
                   this%op_cxx_pptr, &
                   da_f_in_cxx_pptr, &
                   da_dfdt_in_cxx_pptr, &
                   da_lambda_ohms_law%get_cxx_pointer(), &
                   da_b_ohms_law%get_cxx_pointer())

        call profiler_inject("reduction", ierr)
        call profiler_inject_allreduce(ierr)
        call profiler_inject("store", ierr)

        call this%perf_counter%end_measurement()
    end subroutine apply_gpu

    module subroutine finalize_gpu(this)
        type(op_mom_ohms_law_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on the device memory
        ierr = cbind_op_mom_ohms_law_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine finalize_gpu

end submodule op_mom_ohms_law_gpu_s
