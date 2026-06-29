submodule (op_rhs_vlasov_eq_static_m) op_rhs_vlasov_eq_static_gpu_s
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only: C_INT32_T, c_loc
    use params_mesh_m, only: get_use_bsg
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_vlasov_static_initialize( &
            mesh_cxx_pptr, op_cxx_pptr) &
            bind(C, name="cbind_op_vlasov_static_initialize")
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_vlasov_static_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_vlasov_static_finalize")
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_vlasov_static_apply( &
            op_cxx_pptr, mesh_cxx_pptr, da_f_in_cxx_pptr, &
            da_phi_in_cxx_pptr, da_A_par_in_cxx_pptr, da_B_par_in_cxx_pptr, &
            da_f_out_cxx_pptr) bind(C, name="cbind_op_vlasov_static_apply")
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in) :: op_cxx_pptr
            type(C_PTR), intent(in) :: mesh_cxx_pptr
            type(C_PTR), intent(in) :: da_f_in_cxx_pptr
            type(C_PTR), intent(in) :: da_phi_in_cxx_pptr
            type(C_PTR), intent(in) :: da_A_par_in_cxx_pptr
            type(C_PTR), intent(in) :: da_B_par_in_cxx_pptr
            type(C_PTR), intent(inout) :: da_f_out_cxx_pptr
        end function
    end interface

contains

    module subroutine initialize_gpu(this, mesh, bsg_op)
        class(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op

        integer :: ierr

        this%n_flops = 677   ! We estimate division with 4
        this%n_ls    = 266

        if (get_use_bsg()) then
            this%bsg_op => bsg_op
        else
            nullify(this%bsg_op)
        end if

        this%mesh => mesh

        ! Initialize operator member arrays on the device memory
        ierr = cbind_op_vlasov_static_initialize(this%mesh%get_cxx_pointer(), &
                                                 this%op_cxx_pptr)
        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

    end subroutine

    module subroutine apply_gpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                da_B_par_in, da_f_out)
        class(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in
        class(data_array_2d_t), intent(in) :: da_A_par_in
        class(data_array_2d_t), intent(in) :: da_B_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: ierr

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        ierr = cbind_op_vlasov_static_apply( &
                   this%op_cxx_pptr, &
                   this%mesh%get_cxx_pointer(), &
                   da_f_in%get_readonly_cxx_pointer(), &
                   da_phi_in%get_readonly_cxx_pointer(), &
                   da_A_par_in%get_readonly_cxx_pointer(), &
                   da_B_par_in%get_readonly_cxx_pointer(), &
                   da_f_out%get_cxx_pointer())

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_gpu(this)
        type(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on the device memory
        ierr = cbind_op_vlasov_static_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
