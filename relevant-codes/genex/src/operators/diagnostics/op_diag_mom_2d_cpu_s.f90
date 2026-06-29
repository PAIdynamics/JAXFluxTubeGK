submodule (op_diag_mom_2d_m) op_diag_mom_2d_cpu_s
    use mpi
    use, intrinsic :: iso_fortran_env
    use params_species_m, only: get_mass, get_charge, get_temp_scaling, &
                                n_spec_supported
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_T_ref, &
                                      get_c_ref, get_n_ref, get_m_ref
    use params_mesh_m, only: get_n_points_sp
    use params_diagnostics_m, only: get_diagnose_tpc
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use bsg_types_m, only: unpack_bflag
    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
        class(op_diag_mom_2d_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op

        integer :: n, n_points_sp

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        this%absB => mesh%get_absB_pointer()
        this%curl_normb_y => mesh%get_curl_normb_y_pointer()
        this%vp_bsg => bsg_op%get_vp_pointer()
        this%vpw_bsg => bsg_op%get_vpw_pointer()
        this%bsg_flags => mesh%get_bsg_flags_pointer()
        this%mu => mesh%get_mu_pointer()
        this%muw => mesh%get_muw_pointer()
        if (get_diagnose_tpc()) then
            this%loss_cone => mesh%get_loss_cone_pointer()
        endif

        this%n_flops = 29
        this%n_ls = 16
        n_points_sp = get_n_points_sp()

        allocate(this%prefac_bps(n_points_sp))
        allocate(this%prefac_vth(n_points_sp))
        allocate(this%prefac_energy(n_points_sp))
        allocate(this%prefac_heat_flux(n_points_sp))
        allocate(this%prefac_diam(n_points_sp))

        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
            this%prefac_vth(n) = sqrt(2.0_GP * get_temp_scaling(n) &
                                             / get_mass(n))
            this%prefac_energy(n) = get_temp_scaling(n)
            this%prefac_heat_flux(n) = this%prefac_vth(n) &
                                     * get_temp_scaling(n)**1.5_GP
            this%prefac_diam(n) = get_rho_ref() &
                                / (get_charge(n) * get_L_ref()) &
                                * get_temp_scaling(n)**2.0_GP
        enddo

        call this%op_set_uniform%initialize()

    end subroutine

    module subroutine apply_cpu(this, da_f, da_moments, diagnose_tpc)
        class(op_diag_mom_2d_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f
        class(data_array_4d_t), intent(inout) :: da_moments
        logical, optional, intent(in) :: diagnose_tpc

        real(kind=GP) :: bps, b_mu, vp2, vp4, mu2, muvp2
        ! B parallel star, absB times mu, v parallel squared,
        ! v parallel pow 4, mu times v parallel squared
        integer :: ierr
        integer :: lb_stripped(5), ub_stripped(5), i, k, l, m, n
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: f
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        real(kind=GP), pointer, dimension(:,:) :: is_compute
        real(kind=GP), dimension(:,:,:,:), allocatable :: send_buffer
        logical :: diagnose_tpc_local
        integer :: nb_point

        lb_stripped = da_f%get_lbound_stripped()
        ub_stripped = da_f%get_ubound_stripped()

        f => da_f%get_readonly_pointer_stripped()
        moments => da_moments%get_pointer()
        is_compute => this%mesh%get_is_compute_pointer()

        this%n_iterations = da_f%get_size_stripped()

        call this%perf_counter%start_measurement()
        allocate(send_buffer(lb_stripped(1):ub_stripped(1), &
                             lb_stripped(2):ub_stripped(2), 8, &
                             lb_stripped(5):ub_stripped(5)))
        call this%op_set_uniform%apply(send_buffer, 0.0_GP)

        ! Optional TPC default false
        diagnose_tpc_local = .false.
        if (present(diagnose_tpc)) then
            diagnose_tpc_local = diagnose_tpc
        endif

        call profiler_start("reduction", ierr)

        !$omp parallel default (none) &
        !$omp firstprivate(lb_stripped, ub_stripped, diagnose_tpc_local) &
        !$omp shared(f, this, send_buffer, is_compute) &
        !$omp private(i, k, l, m, n, bps, vp2, b_mu, mu2, muvp2, vp4, nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do simd schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            nb_point = unpack_bflag(this%bsg_flags(i))

            ! In the TPC diagnostic we cycle points inside the loss cone
            if (diagnose_tpc_local) then
                if (this%mu(m) * this%loss_cone(i, k) &
                    <= this%vp_bsg(nb_point)%array(l)**2) cycle
            endif

            bps = this%vpw_bsg(nb_point)%array(l) * this%muw(m) * PI &
                              * (this%absB(i, k) + this%prefac_bps(n) &
                                  * this%vp_bsg(nb_point)%array(l) &
                                  * this%curl_normb_y(i, k))
            vp2   = this%vp_bsg(nb_point)%array(l)**2
            b_mu  = this%absB(i, k) * this%mu(m)
            vp4   = vp2 * vp2
            muvp2 = this%mu(m) * vp2
            mu2   = this%mu(m) * this%mu(m)

            ! Density n
            send_buffer(i, k, 1, n) = send_buffer(i, k, 1, n) &
                                    + bps * f(i, k, l, m, n) * is_compute(i, k)
            ! Parallel flow u_par
            send_buffer(i, k, 2, n) = send_buffer(i, k, 2, n) &
                                    + bps * this%prefac_vth(n) &
                                      * this%vp_bsg(nb_point)%array(l) &
                                          * f(i, k, l, m, n) * is_compute(i, k)
            ! Parallel energy E_par
            send_buffer(i, k, 3, n) = send_buffer(i, k, 3, n) &
                                    + bps * this%prefac_energy(n) * vp2 &
                                          * f(i, k, l, m, n) * is_compute(i, k)
            ! Perpendicular energy E_perp
            send_buffer(i, k, 4, n) = send_buffer(i, k, 4, n) &
                                    + bps * this%prefac_energy(n) * b_mu &
                                          * f(i, k, l, m, n) * is_compute(i, k)
            ! Parallel heat flux Q_par
            send_buffer(i, k, 5, n) = send_buffer(i, k, 5, n) &
                                    + bps * this%prefac_heat_flux(n) * vp2 &
                                          * this%vp_bsg(nb_point)%array(l) &
                                          * f(i, k, l, m, n) &
                                          * is_compute(i, k)
            ! Perpendicular heat flux Q_perp
            send_buffer(i, k, 6, n) = send_buffer(i, k, 6, n) &
                                    + bps * this%prefac_heat_flux(n) &
                                          * this%vp_bsg(nb_point)%array(l) &
                                          * b_mu * f(i, k, l, m, n) &
                                          * is_compute(i, k)

            ! Parallel diamagnetic moment K_par
            send_buffer(i, k, 7, n) = send_buffer(i, k, 7, n) &
                                    + bps * this%prefac_diam(n) &
                                          * 2.0_GP &
                                          * (vp4 / this%absB(i, k) + muvp2) &
                                          * f(i, k, l, m, n) &
                                          * is_compute(i, k)

            ! Perpendicular diamagnetic moment K_perp
            send_buffer(i, k, 8, n) = send_buffer(i, k, 8, n) &
                                    + bps * this%prefac_diam(n) &
                                          * (muvp2 / this%absB(i, k) + mu2) &
                                          * f(i, k, l, m, n) &
                                          * is_compute(i, k)

        enddo
        !$omp end do simd nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("reduction", ierr)

        call profiler_start_allreduce(this%dcomm_handler%get_comm_vp_mu(), ierr)

        call MPI_Allreduce(send_buffer, moments, size(send_buffer), MPI_GP, &
                           MPI_SUM, this%dcomm_handler%get_comm_vp_mu(), ierr)

        call profiler_stop_allreduce(ierr)
        call this%perf_counter%end_measurement()
    end subroutine

end submodule
