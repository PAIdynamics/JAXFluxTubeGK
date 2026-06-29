submodule(op_solve_qn_eq_m) op_solve_qn_eq_s
    use mpi
    use genex_fortran_env_m, only: GP_EPS, MPI_GP
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_WRN_FIELD_SOLVE
    use logger_m, only: logger_get_debug_channel
    use profiler_m, only: profiler_start, profiler_stop
    use params_species_m, only: get_mass
    use params_normalization_m, only: get_rho_ref, get_L_ref
    use params_gyrokinetic_system_m, only: get_with_nlin_polarization
    use params_gpu_offload_m, only: get_use_parallax_gpu_offload, &
                                    get_use_parallax_gpu_data_explicit
    use op_copy_m, only: op_copy_cpu_t
    ! From PARALLAX
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cpu_t
#ifdef ENABLE_PARALLAX_GPU
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cxx_t
#endif

    implicit none

contains

    module subroutine initialize(this, dcomm_handler, mesh)
        class(op_solve_qn_eq_t), intent(inout) :: this
        type(dcomm_handler_t), target, intent(inout) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            co_initial, lambda, xi, absB, jacobian
        integer, contiguous, pointer, dimension(:,:) :: not_in_target
        integer, contiguous, pointer, dimension(:,:) :: RZ_indices
        real(kind=GP) :: rho_ref, L_ref
        integer, dimension(2) :: lb, ub, lb_stripped, ub_stripped
        integer :: i, ir, j, k

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh

        RZ_indices => this%mesh%get_RZ_indices_pointer()
        absB => this%mesh%get_absB_pointer()
        jacobian => this%mesh%get_jacobian_pointer()
        not_in_target => this%mesh%get_not_in_target_pointer()
        rho_ref = get_rho_ref()
        L_ref   = get_L_ref()

        do j = 1, 2
            lb(j) = this%dcomm_handler%get_lbound(j)
            ub(j) = this%dcomm_handler%get_ubound(j)
            lb_stripped(j) = this%dcomm_handler%get_lbound_stripped(j)
            ub_stripped(j) = this%dcomm_handler%get_ubound_stripped(j)
        enddo

        allocate(this%da_co_initial)
        allocate(this%da_lambda)
        allocate(this%da_xi)
        call this%da_co_initial%initialize([lb(1), lb_stripped(2)], &
                                           [ub(1), ub_stripped(2)])
        call this%da_lambda%initialize(lb_stripped, ub_stripped)
        call this%da_xi%initialize(lb_stripped, ub_stripped)
        co_initial => this%da_co_initial%get_pointer()
        lambda => this%da_lambda%get_pointer()
        xi => this%da_xi%get_pointer()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) private(i, k, ir) &
        !$omp shared(rho_ref, L_ref, not_in_target, jacobian, absB, &
        !$omp        co_initial, xi, lambda, RZ_indices)
        do k = lb_stripped(2), ub_stripped(2)
            !$omp do simd schedule(static)
            do i = lb_stripped(1), ub_stripped(1)
                ir = RZ_indices(i, k)
                if(not_in_target(ir, k) == 1) then
                    xi(i, k)     = 1.0_GP / jacobian(ir, k)
                    lambda(i, k) = 0.0_GP
                else
                    xi(i, k)     = 0.0_GP
                    lambda(i, k) = 1.0_GP
                endif
            end do
            !$omp end do simd
            !$omp do simd schedule(static)
            do i = lb(1), ub(1)
                co_initial(i, k) = (rho_ref / L_ref)**2 &
                                 / absB(i, k)**2 * jacobian(i, k)
            enddo
            !$omp end do simd
        enddo
        !$omp end parallel

        call this%da_co_initial%update_device()
        call this%da_lambda%update_device()
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
                                               co=co_initial, &
                                               lambda=lambda, &
                                               xi=xi, &
                                               solvers=this%solvers)

        if(get_with_nlin_polarization()) then
            this%is_initialized = .true.
        else
            this%is_initialized = .false.
        endif

    end subroutine

    module subroutine apply(this, da_co_qn_eq, da_b_qn_eq, da_es_pot, residuum)
        class(op_solve_qn_eq_t), target, intent(inout) :: this
        class(data_array_2d_t), intent(in) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_es_pot
        real(kind=GP), intent(out) :: residuum

        integer :: k, size_RZ_plane, iter, ierr, &
                   lb_phi_stripped, ub_phi_stripped
        real(kind=GP) :: loc_residuum, send_residuum
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            co_initial, lambda, xi, co_qn_eq, b_qn_eq, es_pot, &
            co_initial_hptr, co_qn_eq_hptr
        type(op_copy_cpu_t) :: op_copy

        lb_phi_stripped = this%dcomm_handler%get_lbound_stripped(2)
        ub_phi_stripped = this%dcomm_handler%get_ubound_stripped(2)

        call assign_array_pointers()

        if(.not. get_with_nlin_polarization()) then
            ! Update co buffer on the first call
            if(.not. this%is_initialized) then
                ! TODO: Handle the case with device pointer

                ! Assign host pointers to buffer pointers for copy
                co_initial_hptr => this%da_co_initial%get_pointer()
                co_qn_eq_hptr => da_co_qn_eq%get_readonly_pointer()

                call op_copy%apply(co_initial_hptr, co_qn_eq_hptr)
                call this%da_co_initial%update_device()
                this%is_initialized = .true.
            endif

            co_qn_eq => co_initial
        endif

        ! Perform the field solve
        send_residuum = 0.0_GP
        do k = lb_phi_stripped, ub_phi_stripped
            size_RZ_plane = this%mesh%size_RZ_plane(k)

            ! Update the solvers every timestep because co changes
            call profiler_start("update", ierr)
            call this%solvers(k)%update(co=co_qn_eq(:, k), &
                                        lambda=lambda(:, k), &
                                        xi=xi(:, k))
            call profiler_stop("update", ierr)

            call profiler_start("solve", ierr)
            call this%solvers(k)%solve(rhs=b_qn_eq(:size_RZ_plane, k), &
                                       sol=es_pot(:size_RZ_plane, k), &
                                       res=loc_residuum, info=iter)
            call profiler_stop("solve", ierr)

            ! If iter < 0 then the solve was not successful. For direct &
            ! solver iter = 0 is successful.
            if(iter < 0) then
                call handle_error("Quasi-Neutrality solver did not converge!", &
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
        ! Log information about the multigrid to the debug channel
        if(this%dcomm_handler%is_master()) then
            write(logger_get_debug_channel(), *) &
                "    Quasi-Neutrality solver residuum: ", residuum
            if(iter > 0) then
                write(logger_get_debug_channel(), *) &
                    "    Quasi-Neutrality solver iterations: ", iter
            endif
        endif

    contains

        subroutine assign_array_pointers()
            !! Assigns local array pointers to device pointer of the arrays
            !! if explicit GPU data management in PARALLAX is used.
            !! Otherwise, they are assigned to the host pointer.

            if(get_use_parallax_gpu_data_explicit()) then
                co_initial => this%da_co_initial%get_device_pointer()
                lambda => this%da_lambda%get_device_pointer()
                xi => this%da_xi%get_device_pointer()
                co_qn_eq => da_co_qn_eq%get_readonly_device_pointer()
                b_qn_eq => da_b_qn_eq%get_device_pointer()
                es_pot => da_es_pot%get_device_pointer()
            else
                co_initial => this%da_co_initial%get_pointer()
                lambda => this%da_lambda%get_pointer()
                xi => this%da_xi%get_pointer()
                co_qn_eq => da_co_qn_eq%get_readonly_pointer()
                b_qn_eq => da_b_qn_eq%get_pointer()
                es_pot => da_es_pot%get_pointer()
            endif
        end subroutine

    end subroutine

end submodule
