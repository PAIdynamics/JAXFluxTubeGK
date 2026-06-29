submodule (op_mom_maxwells_eq_m) op_mom_maxwells_eq_cpu_s
    use MPI
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_beta_ref
    use, intrinsic :: iso_fortran_env, only: INT64
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use bsg_types_m, only: vspace_bsg_t, unpack_bflag

    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
        class(op_mom_maxwells_eq_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op

        integer :: n, n_points_sp

        this%mesh => mesh
        this%dcomm_handler => dcomm_handler
        this%bsg_op => bsg_op

        n_points_sp = mesh%size_sp()
        allocate(this%prefac_bps(n_points_sp))
        allocate(this%prefac_b_qn(n_points_sp))
        allocate(this%prefac_b_amp(n_points_sp))
        allocate(this%prefac_co_qn(n_points_sp))
        allocate(this%prefac_b_bpar(n_points_sp))

        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
            this%prefac_b_qn(n) = get_charge(n)
            this%prefac_b_amp(n) = get_beta_ref() * get_charge(n) &
                                 * sqrt(get_temp_scaling(n) / 2.0_GP &
                                                            / get_mass(n))
            this%prefac_co_qn(n) = (get_rho_ref() / get_L_ref())**2 &
                                 * get_mass(n)
            this%prefac_b_bpar(n) = -get_beta_ref() / 2.0_GP &
                                  * get_temp_scaling(n)
        enddo

        call this%op_set_uniform%initialize()
    end subroutine

    module subroutine apply_cpu(this, da_f_in, da_co_qn_eq, da_b_qn_eq, &
                                da_b_amps_law, da_b_bpar_eq)
        class(op_mom_maxwells_eq_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_bpar_eq

        integer :: i, k, l, m, n, ierr
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP) :: bps, area
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in
        real(kind=GP), contiguous, pointer, dimension(:,:) :: co_qn_eq, &
                                                              b_qn_eq, &
                                                              b_amps_law, &
                                                              b_bpar_eq
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu, vpw, muw
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, jacobian, &
                                                              curl_normb_y
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        real(kind=GP), allocatable, dimension(:,:,:) :: send_buffer, &
                                                        receive_buffer
        integer, contiguous, pointer, dimension(:) :: bsg_flags_ptr
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: vp_bsg_ptr
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: vpw_bsg_ptr
        integer :: nb_point

        lb = da_f_in%get_lbound()
        ub = da_f_in%get_ubound()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in       => da_f_in%get_readonly_pointer()
        co_qn_eq   => da_co_qn_eq%get_pointer()
        b_qn_eq    => da_b_qn_eq%get_pointer()
        b_amps_law => da_b_amps_law%get_pointer()

        bsg_flags_ptr => this%mesh%get_bsg_flags_pointer()
        vp_bsg_ptr    => this%bsg_op%get_vp_pointer()
        vpw_bsg_ptr   => this%bsg_op%get_vpw_pointer()

        b_bpar_eq  => da_b_bpar_eq%get_pointer()

        allocate(send_buffer(lb(1):ub(1), lb_stripped(2):ub_stripped(2), 4))
        allocate(receive_buffer, mold=send_buffer)
        call this%op_set_uniform%apply(send_buffer, 0.0_GP)

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        absB         => this%mesh%get_absB_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()
        jacobian     => this%mesh%get_jacobian_pointer()
        mu           => this%mesh%get_mu_pointer()
        muw          => this%mesh%get_muw_pointer()
        is_compute   => this%mesh%get_is_compute_pointer()

        call profiler_start("reduction", ierr)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, lb, ub) &
        !$omp private(bps, area, i, k, l, m, n, nb_point) &
        !$omp shared(f_in, this, absB, jacobian, curl_normb_y, &
        !$omp        mu, muw, send_buffer, vp_bsg_ptr, &
        !$omp        vpw_bsg_ptr, bsg_flags_ptr)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)

            nb_point = unpack_bflag(bsg_flags_ptr(i))

            bps = absB(i, k) + this%prefac_bps(n) &
                             * vp_bsg_ptr(nb_point)%array(l) &
                             * curl_normb_y(i, k)
            area = vpw_bsg_ptr(nb_point)%array(l) * muw(m) * bps * PI

            ! Calculate co for the quasi-neutrality equation
            send_buffer(i, k, 1) = send_buffer(i, k, 1) &
                                 + this%prefac_co_qn(n) &
                                   * f_in(i, k, l, m, n) * area &
                                   * (jacobian(i, k) / absB(i, k)**2)

            ! Calculate b for the quasi-neutrality equation
            send_buffer(i, k, 2) = send_buffer(i, k, 2) &
                                 + this%prefac_b_qn(n) &
                                   * f_in(i, k, l, m, n) * area

            ! Calculate b for Ampere's law
            send_buffer(i, k, 3) = send_buffer(i, k, 3) &
                                 + this%prefac_b_amp(n) &
                                 * vp_bsg_ptr(nb_point)%array(l) &
                                   * f_in(i, k, l, m, n) * area

            ! Calculate b for bpar equation
            send_buffer(i, k, 4) = send_buffer(i, k, 4) &
                                 + this%prefac_b_bpar(n) * mu(m) &
                                   * f_in(i, k, l, m, n) * area

        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("reduction", ierr)

        call profiler_start_allreduce(this%dcomm_handler%get_comm_vp_mu_sp(), &
                                      ierr)

        call MPI_Allreduce(send_buffer, receive_buffer, &
                           size(send_buffer), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        call profiler_stop_allreduce(ierr)

        call profiler_start("store", ierr)

        ! Copy information from MPI buffers into the output array.
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, lb, ub) &
        !$omp private(i, k) &
        !$omp shared(this, co_qn_eq, b_amps_law, b_qn_eq, b_bpar_eq, &
        !$omp        is_compute, receive_buffer)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            co_qn_eq(i, k)   = receive_buffer(i, k, 1)
            b_bpar_eq(i, k)  = receive_buffer(i, k, 4)
            ! NOTE: We leave the values unchanged on the ghost points, and in
            !       the target for b, since these values are already set by the
            !       boundary condition operators
            if(is_compute(i, k) == 1.0_GP) then
                b_qn_eq(i, k)    = receive_buffer(i, k, 2)
                b_amps_law(i, k) = receive_buffer(i, k, 3)
            endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        call profiler_stop("store", ierr)

        call this%perf_counter%end_measurement()

        deallocate(send_buffer)
        deallocate(receive_buffer)
    end subroutine

end submodule op_mom_maxwells_eq_cpu_s
