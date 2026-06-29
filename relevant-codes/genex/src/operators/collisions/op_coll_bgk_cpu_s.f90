submodule(op_coll_m) op_coll_bgk_cpu_s
    !! Contains implementation of op_coll_bgk_cpu_t which represents the BGK
    !! collision operator.

    use, intrinsic :: iso_fortran_env, only: INT64
    use math_m, only: PI
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_coll_ref
    use params_collisions_m, only: get_relx_type

    implicit none

contains

    module subroutine initialize_coll_bgk_cpu(this, dcomm_handler, mesh)
        class(op_coll_bgk_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: lb_stripped(5), ub_stripped(5)
        ! Lower and upper bounds of the domain
        integer :: n, n_sp

        this%n_flops = 23 ! We estimate exp with 10 and division with 4
        this%n_ls    = 5

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        ! Check relaxation type
        if(get_relx_type() /= "et" .and. get_relx_type() /= "em") then
            call handle_error("Selected relaxation type "//get_relx_type() &
                              //" is not supported for BGK collisions!", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        this%prefac_nu = get_coll_ref()
        n_sp = mesh%size_sp()

        ! Calculate prefactors for bps
        allocate(this%prefac_bps(n_sp))
        do n = 1, n_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
        enddo

        lb_stripped = this%dcomm_handler%get_lbound_stripped()
        ub_stripped = this%dcomm_handler%get_ubound_stripped()

        allocate(this%da_collog)
        call this%da_collog%initialize(lb_stripped(1:2), ub_stripped(1:2))

    end subroutine

    module subroutine apply_coll_bgk_cpu(this, da_f_in, da_moments, da_f_out)
        class(op_coll_bgk_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: alpha, beta, m, l, k, i
        ! Loop indices
        integer :: n_sp, lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        ! Total number of species, lower and upper bounds of the domain
        real(kind=GP) :: Mfac_ab, prefac_full_nu, prefac_velpart_T
        ! Prefactor for Maxwellian, collision freq. and temperature
        real(kind=GP) :: prefac_relx_u_a, prefac_relx_u_b, &
                         prefac_relx_T_a, prefac_relx_T_b, &
                         delta_u_ab, delta_T_ab
        ! Prefactors for u and T that depend on the relaxation type chosen
        real(kind=GP) :: sum_m, prod_m, sqrt_prod_m, &
                         prefac_vth_alpha, prefac_vth_beta, m_alpha, m_beta, &
                         temp_scaling_alpha, temp_scaling_beta
        ! Prefactors involving masses
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in, f_out
        ! Pointer to the input and output distributions
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        ! Pointer to the output moments
        real(kind=GP), contiguous, pointer, dimension(:,:) :: collog
        ! Pointer to the Coulomb logarithm
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu
        ! Pointer to vp and mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, &
                                                              curl_normb_y
        ! Pointer to absB and curl_normb_y
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        ! Pointer to compute point mask
        real(kind=GP) :: nu, u_ab, T_ab
        ! Collision frequency, mixed drift velocity and mixed temperature
        real(kind=GP) :: bps
        ! B parallel star
        real(kind=GP) :: maxw
        ! Maxwellian

        ! This subroutine applies the operator to the distribution function. It
        ! calculates the moments of f_in, the collision frequency of species
        ! alpha with species beta and the corresponding collision operator for
        ! alpha-beta collisions.

        n_sp = this%mesh%size_sp()
        lb = da_f_in%get_lbound()
        ub = da_f_in%get_ubound()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer()
        moments => da_moments%get_pointer()
        f_out => da_f_out%get_pointer()

        ! First time calculation of Coulomb logarithm
        if(.not. this%is_initialized) then
            call calc_collog(da_moments, this%da_collog)
            this%is_initialized = .true.
        endif
        collog => this%da_collog%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped() * int(n_sp, kind=INT64)
        call this%perf_counter%start_measurement()

        vp           => this%mesh%get_vp_pointer()
        mu           => this%mesh%get_mu_pointer()
        absB         => this%mesh%get_absB_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()
        is_compute   => this%mesh%get_is_compute_pointer()

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, n_sp) &
        !$omp shared(this, f_in, f_out, moments, collog, mu, vp, absB, &
        !$omp        is_compute, curl_normb_y) &
        !$omp private(i, k, l, m, alpha, beta, &
        !$omp         m_alpha, m_beta, sum_m, prod_m, &
        !$omp         temp_scaling_alpha, temp_scaling_beta, &
        !$omp         sqrt_prod_m, prefac_vth_alpha, prefac_vth_beta, &
        !$omp         prefac_full_nu, prefac_velpart_T, Mfac_ab, nu, u_ab, &
        !$omp         T_ab, bps, prefac_relx_u_a, prefac_relx_u_b, &
        !$omp         prefac_relx_T_a, prefac_relx_T_b, maxw, &
        !$omp         delta_u_ab, delta_T_ab)
        ! Loop over all species combinations
        do alpha = lb_stripped(5), ub_stripped(5)
        ! This has to go explicitly through all species because of mpi processes
        ! may be parallized on the species dimension
        do beta = 1, n_sp

            ! Precalculate for innermost loops
            m_alpha = get_mass(alpha)
            m_beta  = get_mass(beta)
            temp_scaling_alpha = get_temp_scaling(alpha)
            temp_scaling_beta  = get_temp_scaling(beta)
            prefac_vth_alpha = sqrt(temp_scaling_alpha / m_alpha)
            prefac_vth_beta = sqrt(temp_scaling_beta / m_beta)
            sum_m       = m_alpha + m_beta
            prod_m      = m_alpha * m_beta
            sqrt_prod_m = sqrt(prod_m)

            ! Prefac nu with masses and charges
            prefac_full_nu = this%prefac_nu * sqrt_prod_m &
                           * (get_charge(alpha) * get_charge(beta))**2

            ! Prefac of mixing temperature with masses
            prefac_velpart_T = (1.0_GP / 3.0_GP) * (prod_m / sum_m) &
                             / get_temp_scaling(alpha)

            ! Depending on relaxation type we adjust the prefactors that are
            ! multiplied on nu, u and T. We do this before the main loop to
            ! avoid conditionals there.
            select case(get_relx_type())
                ! Temperature relaxation
                case("et")
                    !prefac_full_nu does not change in et
                    prefac_relx_u_a = m_alpha / sum_m
                    prefac_relx_u_b = m_beta / sum_m
                    prefac_relx_T_a = 0.5_GP
                    prefac_relx_T_b = 0.5_GP
                ! Momentum relaxation
                case("em")
                    prefac_full_nu  = 0.5_GP * prefac_full_nu &
                                    * sum_m / m_alpha
                    prefac_relx_u_a = 0.5_GP
                    prefac_relx_u_b = 0.5_GP
                    prefac_relx_T_a = m_beta / sum_m
                    prefac_relx_T_b = m_alpha / sum_m
            end select
            delta_u_ab = prefac_vth_beta / prefac_vth_alpha
            delta_T_ab = temp_scaling_beta / temp_scaling_alpha
            prefac_relx_u_b = prefac_relx_u_b * delta_u_ab
            prefac_relx_T_b = prefac_relx_T_b * delta_T_ab

            do m = lb_stripped(4), ub_stripped(4)
            do l = lb_stripped(3), ub_stripped(3)
            do k = lb_stripped(2), ub_stripped(2)
            !$omp do schedule(static)
            do i = lb_stripped(1), ub_stripped(1)

                ! Collision frequency
                nu = prefac_full_nu * collog(i, k) &
                   * moments(i, k, 1, beta) &
                   * (m_alpha * moments(i, k, 3, beta) * temp_scaling_beta &
                     + m_beta * moments(i, k, 3, alpha) * temp_scaling_alpha &
                     )**(-1.5_GP)

                ! Maxwellian parameter: mixing flow
                u_ab = prefac_relx_u_a * moments(i, k, 2, alpha) &
                     + prefac_relx_u_b * moments(i, k, 2, beta)

                ! Maxwellian parameter: mixing temperature
                T_ab = prefac_relx_T_a * moments(i, k, 3, alpha) &
                     + prefac_relx_T_b * moments(i, k, 3, beta) &
                     + prefac_velpart_T &
                        * (moments(i, k, 2, beta) * prefac_vth_beta &
                          - moments(i, k, 2, alpha) * prefac_vth_alpha)**2

                ! Prefactor for Maxwellian
                Mfac_ab = moments(i, k, 1, alpha) * (PI * T_ab)**(-1.5_GP)

                ! B parallel star
                bps = absB(i, k) &
                    + this%prefac_bps(alpha) * vp(l) * curl_normb_y(i, k)

                ! Calculate collision operator: nu * (Maxwellian - f_in) and
                ! add to ouput distribution function
                maxw = Mfac_ab * (absB(i, k) / bps) &
                     * exp(-((vp(l) - u_ab)**2 + absB(i, k) * mu(m)) / T_ab)

                f_out(i, k, l, m, alpha) = f_out(i, k, l, m, alpha) &
                                         + nu * (maxw - f_in(i, k, l, m, alpha))

                ! Apply compute mask
                f_out(i, k, l, m, alpha) = f_out(i, k, l, m, alpha) &
                                         * is_compute(i, k)
            end do
            !$omp end do nowait
            end do
            end do
            end do
        end do
        end do
        !$omp end parallel

        call this%perf_counter%end_measurement()

    end subroutine

end submodule
