submodule(op_bnd_cond_m) op_bnd_cond_neum_gpu_s
    use genex_fortran_env_m, only: CP
    use, intrinsic :: iso_c_binding, only: C_INT32_T, c_loc
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE
    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_bnd_cond_neum_initialize( &
            dcomm_handler_cxx_pptr, mesh_cxx_pptr, rho_center, is_core_ptr, &
            rho_ptr, op_cxx_pptr) &
            bind(C, name="cbind_op_bnd_cond_neum_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_bnd_cond_neum_gpu_t C++ class
            import :: C_PTR, C_INT32_T, CP
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            real(kind=CP), value :: rho_center
            real(kind=CP), dimension(*), intent(inout) :: is_core_ptr
            real(kind=CP), dimension(*), intent(inout) :: rho_ptr
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_bnd_cond_neum_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_bnd_cond_neum_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_bnd_cond_neum_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_bnd_cond_neum_apply( &
            op_cxx_pptr, da_f_inout_cxx_pptr, da_b_qn_eq_cxx_pptr, &
            da_b_amps_law_cxx_pptr, da_b_ohms_law_cxx_pptr) &
            bind(C, name="cbind_op_bnd_cond_neum_apply")
            !! Fortran/C++ interoperable apply routine
            !! for op_bnd_cond_neum_gpu_t
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: op_cxx_pptr
            type(C_PTR), intent(inout) :: da_f_inout_cxx_pptr
            type(C_PTR), intent(inout) :: da_b_qn_eq_cxx_pptr
            type(C_PTR), intent(inout) :: da_b_amps_law_cxx_pptr
            type(C_PTR), intent(inout) :: da_b_ohms_law_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_neum_gpu(this, dcomm_handler, mesh)
        class(op_bnd_cond_neum_gpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: ierr
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_core, rho

        this%mesh => mesh
        is_core   => this%mesh%get_is_core_pointer()
        rho       => this%mesh%get_rho_pointer()

        ! Initialize operator on the device memory
        ierr = cbind_op_bnd_cond_neum_initialize( &
            dcomm_handler%get_cxx_pointer(), &
            this%mesh%get_cxx_pointer(), &
            (this%mesh%rho_max() + this%mesh%rho_min()) / 2.0_GP, &
            is_core, rho, &
            this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        end if
    end subroutine

    module subroutine apply_neum_gpu(this, da_f_inout, da_co_qn_eq, &
                                     da_b_qn_eq, da_b_amps_law, &
                                     da_b_ohms_law, t)
        class(op_bnd_cond_neum_gpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        integer :: ierr

        this%n_iterations = da_f_inout%get_size_stripped()
        call this%perf_counter%start_measurement()

        ierr = cbind_op_bnd_cond_neum_apply(this%op_cxx_pptr, &
                                            da_f_inout%get_cxx_pointer(), &
                                            da_b_qn_eq%get_cxx_pointer(), &
                                            da_b_amps_law%get_cxx_pointer(), &
                                            da_b_ohms_law%get_cxx_pointer())

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_neum_gpu(this)
        type(op_bnd_cond_neum_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on the device memory
        ierr = cbind_op_bnd_cond_neum_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        end if
    end subroutine

end submodule
