submodule (op_diag_mom_0d_m) op_diag_mom_0d_cpu_s
    use MPI
    use, intrinsic :: iso_fortran_env
    use params_species_m, only: n_spec_supported, get_mass, get_charge, &
                                get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_n_ref, &
                                      get_T_ref, get_c_ref, get_m_ref
    use params_mesh_m, only: get_n_points_sp
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use bsg_types_m, only: unpack_bflag
    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
        class(op_diag_mom_0d_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op

        integer :: n, n_points_sp

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        this%absB          => mesh%get_absB_pointer()
        this%normb_tor     => mesh%get_normb_tor_pointer()
        this%curl_normb_y  => mesh%get_curl_normb_y_pointer()
        this%RZw           => mesh%get_RZw_pointer()
        this%jacobian      => mesh%get_jacobian_pointer()
        this%vp_bsg        => bsg_op%get_vp_pointer()
        this%vpw_bsg       => bsg_op%get_vpw_pointer()
        this%mu            => mesh%get_mu_pointer()
        this%phiw          => mesh%get_phiw_pointer()
        this%muw           => mesh%get_muw_pointer()
        this%psi           => mesh%get_psi_pointer()
        this%bsg_flags     => mesh%get_bsg_flags_pointer()

        this%n_flops = 19
        this%n_ls = 4

        n_points_sp = get_n_points_sp()
        allocate(this%prefac_bps(n_points_sp))
        allocate(this%prefac_vth(n_points_sp))
        allocate(this%prefac_energy(n_points_sp))
        allocate(this%charges(n_points_sp))
        allocate(this%prefac_psi(n_points_sp))
        allocate(this%prefac_btor(n_points_sp))

        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
            this%prefac_vth(n) = sqrt(2.0_GP * get_temp_scaling(n) &
                                             / get_mass(n))
            this%prefac_energy(n) = get_temp_scaling(n)
            this%charges(n) = get_charge(n)
            this%prefac_psi(n) = get_charge(n) * get_L_ref() / get_rho_ref()
            this%prefac_btor(n) = sqrt(2.0_GP * get_temp_scaling(n) &
                                              * get_mass(n))
        enddo

    end subroutine

    module subroutine apply_cpu(this, da_f, da_es_pot, da_moments)
        class(op_diag_mom_0d_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f
        class(data_array_2d_t), intent(in) :: da_es_pot
        class(data_array_2d_t), intent(inout) :: da_moments

        real(kind=GP), pointer, dimension(:,:,:,:,:) :: f
        real(kind=GP), pointer, dimension(:,:) :: es_pot
        real(kind=GP), contiguous, pointer, dimension(:,:) :: moments
        real(kind=GP) :: send_buffer(6, n_spec_supported)
        ! NOTE: We use a static send buffer here up to the maximum number of
        !       species supported. This is to avoid allocation in the operator.
        !       We can not allocate the memory in initialization of the derived
        !       type due to the OpenMP reduction.
        real(kind=GP) :: bps, area
        integer :: ierr
        integer :: lb_stripped(5), ub_stripped(5), i, k, l, m, n
        integer :: nb_point

        lb_stripped = da_f%get_lbound_stripped()
        ub_stripped = da_f%get_ubound_stripped()

        f => da_f%get_readonly_pointer_stripped()
        es_pot => da_es_pot%get_readonly_pointer_stripped()
        moments => da_moments%get_pointer()

        this%n_iterations = da_f%get_size_stripped()

        call this%perf_counter%start_measurement()
        send_buffer = 0.0_GP

        call profiler_start("reduction", ierr)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(f, this, es_pot) &
        !$omp private(i, k, l, m, n, bps, area, nb_point) &
        !$omp reduction(+:send_buffer)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            nb_point = unpack_bflag(this%bsg_flags(i))

            ! NOTE: RZw constains the is_compute mask already to make sure
            !       only the correct points are used in the quadrature.
            area = PI * this%RZw(i, k) * this%jacobian(i, k) &
                 * this%phiw(k) * this%vpw_bsg(nb_point)%array(l) * this%muw(m)
            ! Bps with volume element
            bps = area * (this%absB(i, k) + this%prefac_bps(n) &
                         * this%vp_bsg(nb_point)%array(l) &
                         * this%curl_normb_y(i, k))

            ! Density n
            send_buffer(1, n) = send_buffer(1, n) &
                              + bps * f(i, k, l, m, n)
            ! Parallel flow u_par
            send_buffer(2, n) = send_buffer(2, n) &
                              + bps * this%vp_bsg(nb_point)%array(l) &
                                * f(i, k, l, m, n)
            ! Parallel energy E_par
            send_buffer(3, n) = send_buffer(3, n) &
                              + bps * this%vp_bsg(nb_point)%array(l)**2 &
                                * f(i, k, l, m, n)
            ! Perpendicular energy E_perp
            send_buffer(4, n) = send_buffer(4, n) &
                              + bps * this%absB(i, k) * this%mu(m) &
                                    * f(i, k, l, m, n)
            ! Electrostatic energy E_es
            send_buffer(5, n) = send_buffer(5, n) &
                              + 0.5_GP * bps * es_pot(i, k) * f(i, k, l, m, n)
            ! Canonical toroidal momentum p_phi
            send_buffer(6, n) = send_buffer(6, n) &
                + bps * (this%prefac_psi(n) * this%psi(i, k) &
                        + this%prefac_btor(n) * this%normb_tor(i, k) &
                                              * this%jacobian(i, k) &
                                              / this%absB(i, k) &
                                              * this%vp_bsg(nb_point)%array(l) &
                        ) * f(i, k, l, m, n)

        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("reduction", ierr)

        ! Multiply constant common factors
        do n = lb_stripped(5), ub_stripped(5)
            send_buffer(2, n) = send_buffer(2, n) * this%prefac_vth(n)
            send_buffer(3, n) = send_buffer(3, n) * this%prefac_energy(n)
            send_buffer(4, n) = send_buffer(4, n) * this%prefac_energy(n)
            send_buffer(5, n) = send_buffer(5, n) * this%charges(n)
        enddo

        moments = 0.0_GP
        call profiler_start_allreduce(this%dcomm_handler%get_comm_phi_vp_mu(), &
                                      ierr)

        call MPI_Allreduce(send_buffer(:, lb_stripped(5):ub_stripped(5)), &
                           moments, size(moments), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_phi_vp_mu(), ierr)

        call profiler_stop_allreduce(ierr)
        call this%perf_counter%end_measurement()
    end subroutine
end submodule
