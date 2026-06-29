submodule (op_mom_ohms_law_m) op_mom_ohms_law_cpu_s
    use MPI
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use, intrinsic :: iso_fortran_env, only: INT64
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_beta_ref
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use bsg_types_m, only: vspace_bsg_t, unpack_bflag

    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
        class(op_mom_ohms_law_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op
        integer :: n, n_points_sp

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        this%bsg_op => bsg_op

        n_points_sp = mesh%size_sp()
        allocate(this%prefac_bps(n_points_sp))
        allocate(this%prefac_norm_b(n_points_sp))
        allocate(this%prefac_norm_lambda(n_points_sp))
        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
            this%prefac_norm_lambda(n) = get_beta_ref() / 2.0_GP &
                                       * get_charge(n)**2 / get_mass(n)
            this%prefac_norm_b(n) = get_beta_ref() / 2.0_GP * get_charge(n) &
                                  * sqrt(2.0_GP * get_temp_scaling(n) &
                                                / get_mass(n))
        enddo

        call this%op_set_uniform%initialize()
    end subroutine initialize_cpu

    module subroutine apply_cpu(this, da_f_in, da_dfdt_in, da_lambda_ohms_law, &
                                da_b_ohms_law)
        class(op_mom_ohms_law_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_5d_t), intent(in) :: da_dfdt_in
        class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law

        integer :: i, ir, k, l, m, n, ierr
        integer, dimension(5) :: lb_stripped, ub_stripped
        real(kind=GP) :: bps, dfdvp, area
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in, &
                                                                    dfdt_in
        real(kind=GP), contiguous, pointer, dimension(:,:) :: lambda_ohms_law, &
                                                              b_ohms_law
        real(kind=GP), contiguous, pointer, dimension(:) :: muw
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, &
                                                              curl_normb_y
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer, contiguous, pointer, dimension(:,:) :: not_in_target
        real(kind=GP), allocatable, dimension(:,:,:) :: send_buffer, &
                                                        receive_buffer
        integer, contiguous, pointer, dimension(:,:) :: RZ_indices
        integer, contiguous, pointer, dimension(:) :: bsg_flags
        real(kind=GP), contiguous, dimension(:), pointer :: prefac_cd_vp_bsg
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: vp_bsg, vpw_bsg
        integer :: nb_point

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer()
        dfdt_in => da_dfdt_in%get_readonly_pointer()
        lambda_ohms_law => da_lambda_ohms_law%get_pointer()
        b_ohms_law => da_b_ohms_law%get_pointer()

        allocate(send_buffer(lb_stripped(1):ub_stripped(1), &
                             lb_stripped(2):ub_stripped(2), 2))
        allocate(receive_buffer(lb_stripped(1):ub_stripped(1), &
                                lb_stripped(2):ub_stripped(2), 2))

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        absB          => this%mesh%get_absB_pointer()
        curl_normb_y  => this%mesh%get_curl_normb_y_pointer()
        muw           => this%mesh%get_muw_pointer()
        not_in_target => this%mesh%get_not_in_target_pointer()
        is_compute    => this%mesh%get_is_compute_pointer()
        RZ_indices    => this%mesh%get_RZ_indices_pointer()

        bsg_flags => this%mesh%get_bsg_flags_pointer()
        prefac_cd_vp_bsg => this%bsg_op%get_prefac_vp_pointer()
        vp_bsg => this%bsg_op%get_vp_pointer()
        vpw_bsg => this%bsg_op%get_vpw_pointer()

        ! First touch initialize and reset buffer
        call this%op_set_uniform%apply(send_buffer, 0.0_GP)

        call profiler_start("reduction", ierr)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, f_in, dfdt_in, absB, curl_normb_y, muw, &
        !$omp        send_buffer, bsg_flags, prefac_cd_vp_bsg, vp_bsg, &
        !$omp        vpw_bsg) &
        !$omp private(bps, area, dfdvp, i, k, l, m, n, nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            nb_point = unpack_bflag(bsg_flags(i))

            bps = absB(i, k) + this%prefac_bps(n) &
                             * vp_bsg(nb_point)%array(l) &
                             * curl_normb_y(i, k)
            area = vpw_bsg(nb_point)%array(l) * muw(m) * bps * PI

            dfdvp = prefac_cd_vp_bsg(nb_point) * &
                    (- 1.0_GP / 12.0_GP * f_in(i,k,l + 2,m,n) &
                     + 2.0_GP / 3.0_GP  * f_in(i,k,l + 1,m,n) &
                     - 2.0_GP / 3.0_GP  * f_in(i,k,l - 1,m,n) &
                     + 1.0_GP / 12.0_GP * f_in(i,k,l - 2,m,n))

            ! Calculate lambda
            send_buffer(i, k, 1) = send_buffer(i, k, 1) &
                -   this%prefac_norm_lambda(n) * vp_bsg(nb_point)%array(l) &
                    * dfdvp &
                    * area

            ! Calculate b
            send_buffer(i, k, 2) = send_buffer(i, k, 2) &
                +   this%prefac_norm_b(n) * vp_bsg(nb_point)%array(l) &
                    * dfdt_in(i, k, l, m, n) &
                    * area
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

        call MPI_Allreduce(send_buffer, &
                           receive_buffer, &
                           size(send_buffer), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        call profiler_stop_allreduce(ierr)

        call profiler_start("store", ierr)

        ! Copy information from MPI buffers into the output arrays.
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, lambda_ohms_law, b_ohms_law, not_in_target, &
        !$omp        is_compute, receive_buffer, RZ_indices) &
        !$omp private(i, k, ir)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            ! NOTE: We leave the values unchanged on the ghost points, and in
            !       the target for b, since these values are already set by the
            !       boundary condition operators
            if(is_compute(i, k) == 1.0_GP) then
                b_ohms_law(i, k) = receive_buffer(i, k, 2)
            endif

            ir = RZ_indices(i, k)
            if(is_compute(ir, k) == 1.0_GP) then
                lambda_ohms_law(i, k) = receive_buffer(ir, k, 1)
            elseif(not_in_target(ir, k) == 0) then
                ! NOTE: We set the values of lambda to one in the target, since
                !       they are not used by the solver.
                lambda_ohms_law(i, k) = 1.0_GP
            endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        call profiler_stop("store", ierr)

        call this%perf_counter%end_measurement()

        deallocate(send_buffer)
        deallocate(receive_buffer)
    end subroutine apply_cpu

end submodule op_mom_ohms_law_cpu_s
