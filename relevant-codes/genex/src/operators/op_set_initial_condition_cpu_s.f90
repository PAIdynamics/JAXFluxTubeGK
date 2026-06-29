submodule (op_set_initial_condition_m) op_set_initial_condition_cpu_s
    use mpi
    use genex_fortran_env_m, only: MPI_GP, GP_EPS
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use params_species_m, only: get_mass, get_charge, get_is_electrons, &
                                get_temp_scaling
    use params_mesh_m, only: get_n_points_sp
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_initial_condition_m, only: get_initial_perturbation, &
                                          get_use_canonical_maxw
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_types_m, only: unpack_bflag, vspace_bsg_t

    ! From PARALLAX
    use equilibrium_factory_m, only: NUMERICAL
    use descriptors_m, only: DISTRICT_PRIVFLUX, DISTRICT_SOL, DISTRICT_WALL, &
                             DISTRICT_DOME, DISTRICT_OUT
    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, &
                                     dist_initial_container, bsg_op)
        class(op_set_initial_condition_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(in) :: mesh
        type(dist_initial_container_t), &
            dimension(:), allocatable, intent(in) :: dist_initial_container
        type(bsg_operators_t), target, intent(inout) :: bsg_op
        integer :: n, n_points_sp

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        this%dist_initial_container = dist_initial_container
        this%bsg_op => bsg_op

        n_points_sp = get_n_points_sp()
        allocate(this%prefac_bps(n_points_sp))
        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
        enddo

        this%n_flops = 0 ! we can not reliably estimate the number of flops
        this%n_ls = 5
    end subroutine

    module subroutine set_mode_perturbation_cpu(this, da_f_inout)
        use params_set_mode_m, only: get_mean_rho, get_sigma_rho, &
                                     get_strength, &
                                     get_poloidal_mode_number, &
                                     get_toroidal_mode_number, get_k_par
        class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), contiguous, pointer, dimension(:) :: phi
        real(kind=GP) :: strength, sigma_rho, mean_rho, k_par
        integer :: poloidal_mode, toroidal_mode
        real(kind=GP), dimension(:,:), pointer :: rho, theta

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()
        phi => this%mesh%get_phi_pointer()
        rho => this%mesh%get_rho_pointer()
        theta => this%mesh%get_theta_pointer()
        strength = get_strength()
        sigma_rho = get_sigma_rho()
        mean_rho = get_mean_rho()
        poloidal_mode = get_poloidal_mode_number()
        toroidal_mode = get_toroidal_mode_number()
        k_par = get_k_par()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp private(n, m, l, k, i) &
        !$omp shared(this, f_inout, strength, mean_rho, sigma_rho, &
        !$omp        poloidal_mode, toroidal_mode, k_par, phi, rho, theta)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            if(n == 1) then
                f_inout(i, k, l, m, n) =  f_inout(i, k, l, m, n) * &
                    (1.0_GP + strength &
                     * exp(- (rho(i, k) - mean_rho)**2 / sigma_rho**2) &
                     * cos(poloidal_mode * theta(i, k) &
                           + k_par * toroidal_mode * phi(k)))
            endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    module subroutine set_blob_perturbation_cpu(this, da_f_inout)
        use params_set_blob_m, only: get_mean_R, get_mean_Z, get_mean_phi, &
                                     get_mean_vp, get_sigma_R, get_sigma_Z, &
                                     get_strength, get_sigma_phi
        class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n, nb_point
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP) :: vp
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), allocatable, dimension(:) :: mean_R_trace, mean_Z_trace
        real(kind=GP), contiguous, pointer, dimension(:) :: phi, mu
        integer, contiguous, pointer, dimension(:) :: bsg_flags
        real(kind=GP), contiguous, pointer, dimension(:,:) :: B, R, Z
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: vp_bsg
        real(kind=GP) :: mean_R, mean_Z, mean_phi, mean_vp, sigma_R, &
                         sigma_Z, sigma_phi, inv_sigma_R, inv_sigma_Z, &
                         inv_sigma_phi, norm_real, norm_vel, vabs2

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()
        B => this%mesh%get_absB_pointer()
        R => this%mesh%get_R_pointer()
        Z => this%mesh%get_Z_pointer()
        phi => this%mesh%get_phi_pointer()
        mu => this%mesh%get_mu_pointer()
        bsg_flags => this%mesh%get_bsg_flags_pointer()
        vp_bsg => this%bsg_op%get_vp_pointer()

        norm_real = get_strength()
        mean_R = get_mean_R()
        mean_Z = get_mean_Z()
        mean_phi = get_mean_phi()
        mean_vp = get_mean_vp()
        sigma_R = get_sigma_R()
        sigma_Z = get_sigma_Z()
        sigma_phi = get_sigma_phi()
        inv_sigma_R = 1.0_GP / sigma_R
        inv_sigma_Z = 1.0_GP / sigma_Z
        inv_sigma_phi = 1.0_GP / sigma_phi

        allocate(mean_R_trace(this%mesh%size_phi()))
        allocate(mean_Z_trace(this%mesh%size_phi()))

        ! Determine center of field-aligned blob on each plane
        do k = 1, this%mesh%size_phi()
            call this%mesh%trace(mean_R, mean_Z, mean_phi, &
                                 (phi(k) - mean_phi), &
                                 mean_R_trace(k), &
                                 mean_Z_trace(k))
        enddo

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(f_inout, B, R, Z, phi, mu, this, mean_vp, &
        !$omp        inv_sigma_phi, inv_sigma_Z, inv_sigma_R, mean_Z_trace, &
        !$omp        mean_R_trace, mean_phi, norm_real, bsg_flags, vp_bsg) &
        !$omp private(n, m, l, k, i, norm_vel, vabs2, nb_point, vp)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do simd schedule(static)
        do i = lb(1), ub(1)

            nb_point = unpack_bflag(bsg_flags(i))
            vp = vp_bsg(nb_point)%array(l)

            !norm_vel = (PI * temp_prof(i, k, n)) ** (-1.5_GP)
            ! NOTE: the blob has constant temperature
            norm_vel = (PI) ** (-1.5_GP)
            vabs2 = mu(m) * B(i, k) + (vp - mean_vp)**2
            f_inout(i, k, l, m, n) =  f_inout(i, k, l, m, n) + &
                + norm_real * norm_vel &
                  !* exp(- vabs2 / temp_prof(i, k, n) &
                  * exp(- vabs2 &
                        - 0.5 * ( &
                          ((R(i, k) - mean_R_trace(k)) * inv_sigma_R)**2 &
                        + ((Z(i, k) - mean_Z_trace(k)) * inv_sigma_Z)**2 &
                        + ((phi(k) - mean_phi) * inv_sigma_phi)**2))
        enddo
        !$omp end do simd nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        deallocate(mean_R_trace)
        deallocate(mean_Z_trace)
    end subroutine

    module subroutine apply_cpu(this, da_f_inout)
        class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n, ierr, nb_point
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        integer, allocatable :: sp_index(:)
        integer, parameter :: electron_index = 1, ion_index = 2
        logical :: use_canonical_maxw
        real(kind=GP) :: vp, vpw
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP) :: rho, bps, area, abs_charge
        real(kind=GP), contiguous, pointer, dimension(:) :: mu, phi, muw
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, is_core, &
                                                              curl_normb_y, &
                                                              rho_buffer, theta
        real(kind=GP), allocatable, dimension(:,:,:) :: send_buffer, &
                                                        charge_density
        type(op_set_uniform_cpu_t) :: op_set_uniform
        type(vspace_bsg_t), contiguous, pointer, dimension(:) :: vp_bsg, vpw_bsg
        integer, contiguous, pointer, dimension(:) :: bsg_flags
        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()

        this%n_iterations = da_f_inout%get_size()
        call this%perf_counter%start_measurement()

        mu           => this%mesh%get_mu_pointer()
        phi          => this%mesh%get_phi_pointer()
        muw          => this%mesh%get_muw_pointer()
        absB         => this%mesh%get_absB_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()
        rho_buffer   => this%mesh%get_rho_pointer()
        theta        => this%mesh%get_theta_pointer()
        is_core      => this%mesh%get_is_core_pointer()
        bsg_flags    => this%mesh%get_bsg_flags_pointer()
        vp_bsg       => this%bsg_op%get_vp_pointer()
        vpw_bsg      => this%bsg_op%get_vpw_pointer()

        ! Set the initial background profile

        ! NOTE: To achieve quasi-neutrality, we calculate the charge densities
        !       of the electrons and ions, and adjust the electron distribution
        !       function at the end
        allocate(send_buffer(lb(1):ub(1), lb_stripped(2):ub_stripped(2), 1:2))
        allocate(charge_density(lb(1):ub(1), &
                 lb_stripped(2):ub_stripped(2), 1:2))
        allocate(sp_index(lb_stripped(5):ub_stripped(5)))

        call op_set_uniform%apply(send_buffer, 0.0_GP)
        call op_set_uniform%apply(charge_density, 0.0_GP)
        do n = lb_stripped(5), ub_stripped(5)
        if(get_is_electrons(n)) then
            sp_index(n) = electron_index
        else
            sp_index(n) = ion_index
        endif
        enddo

        use_canonical_maxw = get_use_canonical_maxw()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, &
        !$omp              use_canonical_maxw, sp_index) &
        !$omp shared(this, f_inout, send_buffer, absB, mu, phi, &
        !$omp        muw, curl_normb_y, rho_buffer, theta, is_core, bsg_flags, &
        !$omp        vp_bsg, vpw_bsg) &
        !$omp private(n, m, l, k, i, rho, bps, area, abs_charge, nb_point, vp, &
        !$omp         vpw)
        do n = lb_stripped(5), ub_stripped(5)
        abs_charge = abs(get_charge(n))
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        ! We set the initial condition also on the ghosts in RZ
        !$omp do schedule(static)
        do i = lb(1), ub(1)

            nb_point = unpack_bflag(bsg_flags(i))
            vp  = vp_bsg(nb_point)%array(l)
            vpw = vpw_bsg(nb_point)%array(l)

            ! Choose ordinary or canonical maxwellian based on parameter
            if(use_canonical_maxw) then
                rho = this%mesh%canonical_rho(i, k, l, m, n)
            else
                rho = rho_buffer(i, k)
            end if
            ! When outside of the closed flux surface region, we set rho to 1.0
            ! for the numerical and 3D equilibria
            if((this%mesh%equi_type() == NUMERICAL &
                .or. .not. this%mesh%is_axisymmetric()) &
                .and. is_core(i, k) == 0.0_GP) then
                rho = 1.0_GP
            endif

            f_inout(i, k, l, m, n) = this%dist_initial_container(n)%eval( &
                                     rho, theta(i, k), phi(k), vp, &
                                     mu(m) * absB(i, k))

            ! Calculate integration area in velocity space
            bps = absB(i, k) + this%prefac_bps(n) * vp * curl_normb_y(i, k)
            area = vpw * muw(m) * bps * PI

            ! Integrate f in the same loop to calculate the charge density
            ! NOTE: We calculate the magnitude of the electron charge density
            send_buffer(i, k, sp_index(n)) = send_buffer(i, k, sp_index(n)) &
                                           + f_inout(i, k, l, m, n) &
                                             * abs_charge * area
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        ! Calculate total charge density
        call MPI_Allreduce(send_buffer, charge_density, &
                           size(send_buffer), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        ! Correct electron distribution function to ensure quasi-neutrality to
        ! machine precision
        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(f_inout, charge_density) &
        !$omp private(n, m, l, k, i)
        do n = lb_stripped(5), ub_stripped(5)
        ! Only for electrons
        if(.not. get_is_electrons(n)) cycle
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ! Only divide if denominator larger than zero
            if(charge_density(i, k, electron_index) > GP_EPS) then
                f_inout(i, k, l, m, n) = f_inout(i, k, l, m, n) &
                                       * charge_density(i, k, ion_index) &
                                       / charge_density(i, k, electron_index)
            end if
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        ! Apply perturbation
        select case(get_initial_perturbation())
            case("noise")
                call this%set_noise_perturbation(da_f_inout)
            case("blob")
                call this%set_blob_perturbation(da_f_inout)
            case("mode")
                call this%set_mode_perturbation(da_f_inout)
        end select

        call this%perf_counter%end_measurement()

    end subroutine

end submodule
