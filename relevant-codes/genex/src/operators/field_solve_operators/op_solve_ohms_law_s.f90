submodule(op_solve_ohms_law_m) op_solve_ohms_law_s
    use mpi
    use genex_fortran_env_m, only: GP_EPS, MPI_GP
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_WRN_FIELD_SOLVE
    use logger_m, only: logger_get_debug_channel
    use profiler_m, only: profiler_start, profiler_stop
    use params_normalization_m, only: get_rho_ref, get_L_ref
    use params_gpu_offload_m, only: get_use_parallax_gpu_offload, &
                                    get_use_parallax_gpu_data_explicit
    ! From PARALLAX
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cpu_t
#ifdef ENABLE_PARALLAX_GPU
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cxx_t
#endif

    implicit none

contains

    module subroutine initialize(this, dcomm_handler, mesh)
        class(op_solve_ohms_law_t), intent(inout) :: this
        type(dcomm_handler_t), target, intent(inout) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        real(kind=GP), allocatable, dimension(:,:) :: lambda
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            co, xi, jacobian
        integer, contiguous, pointer, dimension(:,:) :: not_in_target
        integer, contiguous, pointer, dimension(:,:) :: RZ_indices
        real(kind=GP) :: rho_ref, L_ref
        integer, dimension(2) :: lb, ub, lb_stripped, ub_stripped
        integer :: i, ir, j, k

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh

        RZ_indices => this%mesh%get_RZ_indices_pointer()
        not_in_target => this%mesh%get_not_in_target_pointer()
        jacobian  => this%mesh%get_jacobian_pointer()
        rho_ref = get_rho_ref()
        L_ref   = get_L_ref()

        do j = 1, 2
            lb(j) = this%dcomm_handler%get_lbound(j)
            ub(j) = this%dcomm_handler%get_ubound(j)
            lb_stripped(j) = this%dcomm_handler%get_lbound_stripped(j)
            ub_stripped(j) = this%dcomm_handler%get_ubound_stripped(j)
        enddo

        allocate(this%da_co)
        allocate(this%da_xi)
        call this%da_co%initialize([lb(1), lb_stripped(2)], &
                                   [ub(1), ub_stripped(2)])
        call this%da_xi%initialize(lb_stripped, ub_stripped)
        co => this%da_co%get_pointer()
        xi => this%da_xi%get_pointer()

        ! Local lambda buffer containing preliminary values for
        ! solver initialization
        allocate(lambda(lb_stripped(1):ub_stripped(1), &
                        lb_stripped(2):ub_stripped(2)))

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) private(i, k, ir) &
        !$omp shared(rho_ref, L_ref, not_in_target, jacobian, co, xi, lambda, &
        !$omp        RZ_indices)
        do k = lb_stripped(2), ub_stripped(2)
            !$omp do simd schedule(static)
            do i = lb_stripped(1), ub_stripped(1)
                ir = RZ_indices(i, k)
                if(not_in_target(ir, k) == 1) then
                    xi(i, k)     = (rho_ref / L_ref)**2 / jacobian(ir, k)
                    lambda(i, k) = 0.0_GP
                else
                    xi(i, k)     = 0.0_GP
                    lambda(i, k) = 1.0_GP
                endif
            end do
            !$omp end do simd
            !$omp do simd schedule(static)
            do i = lb(1), ub(1)
                co(i, k) = jacobian(i, k)
            enddo
            !$omp end do simd
        enddo
        !$omp end parallel

        call this%da_co%update_device()
        call this%da_xi%update_device()

        ! Allocate, create and initialize all solvers
        if(get_use_parallax_gpu_offload()) then
#ifdef ENABLE_PARALLAX_GPU
            allocate(helmholtz_solver_mgmres_cxx_t :: &
                     this%solvers(lb_stripped(2):ub_stripped(2)))
#endif
        else
            allocate(helmholtz_solver_mgmres_cpu_t :: &
                     this%solvers(lb_stripped(2):ub_stripped(2)))
        endif

        call this%mesh%create_helmholtz_solver(lb, ub, &
                                               lb_stripped, ub_stripped, &
                                               co=co, &
                                               lambda=lambda, &
                                               xi=xi, &
                                               solvers=this%solvers)

        deallocate(lambda)

    end subroutine

    module subroutine apply(this, da_lambda_ohms_law, da_b_ohms_law, &
                            da_E_par, residuum)
        class(op_solve_ohms_law_t), target, intent(inout) :: this
        class(data_array_2d_t), intent(in) :: da_lambda_ohms_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        class(data_array_2d_t), intent(inout) :: da_E_par
        real(kind=GP), intent(out) :: residuum

        integer :: k, size_RZ_plane, iter, ierr, &
                   lb_phi_stripped, ub_phi_stripped
        real(kind=GP) :: loc_residuum, send_residuum
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            co, xi, lambda_ohms_law, b_ohms_law, E_par

        lb_phi_stripped = this%dcomm_handler%get_lbound_stripped(2)
        ub_phi_stripped = this%dcomm_handler%get_ubound_stripped(2)

        call assign_array_pointers()

        ! Update the solvers and perform the field solve
        send_residuum = 0.0_GP
        do k = lb_phi_stripped, ub_phi_stripped
            size_RZ_plane = this%mesh%size_RZ_plane(k)

            ! Update the solvers every timestep because lambda changes
            call profiler_start("update", ierr)
            call this%solvers(k)%update(co=co(:, k), &
                                        lambda=lambda_ohms_law(:, k), &
                                        xi=xi(:, k))
            call profiler_stop("update", ierr)

            call profiler_start("solve", ierr)
            call this%solvers(k)%solve(rhs=b_ohms_law(:size_RZ_plane, k), &
                                       sol=E_par(:size_RZ_plane, k), &
                                       res=loc_residuum, info=iter)
            call profiler_stop("solve", ierr)

            ! If iter < 0 then the solve was not successful. For direct &
            ! solver iter = 0 is successful.
            if(iter < 0) then
                call handle_error("Ohm's law solver did not converge!", &
                                  GENEX_WRN_FIELD_SOLVE, __LINE__, __FILE__, &
                                  additional_info=error_info_t(&
                                    "Residuum: ", r_info=[loc_residuum]))
            endif
            if(isnan(loc_residuum)) then
                send_residuum = loc_residuum
                exit
            else
                send_residuum = max(send_residuum, loc_residuum)
            endif
        enddo

        ! Communicate the residuum value over all MPI processes

        ! We first check if any residuum is NaN by performing a sum reduction.
        ! If that is the case NaN is returned as the residuum value of the
        ! solve.
        call MPI_Allreduce(send_residuum, residuum, 1, MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_phi(), ierr)
        if(.not. isnan(residuum)) then
            ! If all residuum values are finite we perform a max reduction.
            ! The maximum residuum over all field solves is returned.
            call MPI_Allreduce(send_residuum, residuum, 1, MPI_GP, MPI_MAX, &
                               this%dcomm_handler%get_comm_phi(), ierr)
        endif
        ! Log information about the residuum and the iterations to the
        ! debug channel
        if(this%dcomm_handler%is_master()) then
            write(logger_get_debug_channel(), *) &
                "   Ohms's law solver residuum: ", residuum
            if(iter > 0) then
                write(logger_get_debug_channel(), *) &
                    "    Ohms's law solver iterations: ", iter
            endif
        endif

    contains

        subroutine assign_array_pointers()
            !! Assigns local array pointers to device pointer of the arrays
            !! if explicit GPU data management in PARALLAX is used.
            !! Otherwise, they are assigned to the host pointer.

            if(get_use_parallax_gpu_data_explicit()) then
                co => this%da_co%get_device_pointer()
                xi => this%da_xi%get_device_pointer()
                lambda_ohms_law => &
                    da_lambda_ohms_law%get_readonly_device_pointer()
                b_ohms_law => da_b_ohms_law%get_device_pointer()
                E_par => da_E_par%get_device_pointer()
            else
                co => this%da_co%get_pointer()
                xi => this%da_xi%get_pointer()
                lambda_ohms_law => da_lambda_ohms_law%get_readonly_pointer()
                b_ohms_law => da_b_ohms_law%get_pointer()
                E_par => da_E_par%get_pointer()
            endif
        end subroutine

    end subroutine

end submodule
