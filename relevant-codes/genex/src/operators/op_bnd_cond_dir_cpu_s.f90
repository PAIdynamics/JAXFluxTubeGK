submodule(op_bnd_cond_m) op_bnd_cond_dir_cpu_s
    use math_m, only: PI
    use params_species_m, only: get_mass, get_charge
    use params_normalization_m, only : get_rho_ref, get_L_ref
    implicit none

contains

    module subroutine initialize_dir_cpu(this, dcomm_handler, mesh)
        class(op_bnd_cond_dir_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%mesh => mesh

    end subroutine

    module subroutine apply_dir_cpu(this, da_f_inout, da_co_qn_eq, &
                                    da_b_qn_eq, da_b_amps_law, &
                                    da_b_ohms_law, t)
        class(op_bnd_cond_dir_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        integer :: i, k
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:) :: b_qn_eq, &
                                                              b_amps_law, &
                                                              b_ohms_law
        real(kind=GP), pointer, dimension(:,:) :: is_compute

        ! NOTE: The Dirichlet BC operator is time independent, thus t is not
        !       used. Further, no BC is required for co_qn_eq. The B_par BC
        !       is set in op_mom_maxwell

        this%n_iterations = da_b_qn_eq%get_size_stripped()
        call this%perf_counter%start_measurement()

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        b_qn_eq    => da_b_qn_eq%get_pointer()
        b_amps_law => da_b_amps_law%get_pointer()
        b_ohms_law => da_b_ohms_law%get_pointer()

        is_compute => this%mesh%get_is_compute_pointer()
        ! The compute mask is used to set the boundary condition since its
        ! value is zero for ghosts, fillers, and points in the target, and one
        ! otherwise

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(b_qn_eq, b_amps_law, b_ohms_law, is_compute) &
        !$omp private(i, k)
        do k = lb_stripped(2), ub_stripped(2)
            !$omp do schedule(static)
            do i = lb(1), ub(1)
                b_qn_eq(i, k)    = b_qn_eq(i, k)    * is_compute(i, k)
                b_amps_law(i, k) = b_amps_law(i, k) * is_compute(i, k)
                b_ohms_law(i, k) = b_ohms_law(i, k) * is_compute(i, k)
            end do
            !$omp end do nowait
        end do
        !$omp end parallel

        call this%perf_counter%end_measurement()
    end subroutine

end submodule
