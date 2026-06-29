submodule(op_coll_m) op_coll_lbd_cpu_s
    !! Contains the implementation of op_coll_lbd_cpu_t,
    !! the finite volume implementation of the Lenard-Bernstein/Dougherty
    !! collision operator.

    use MPI
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP_EPS, MPI_GP
    use math_m, only: PI
    use logger_m, only: logger_get_info_channel
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_coll_ref
    use params_collisions_m, only: get_temp_floor, get_relx_type

    implicit none

contains

    module subroutine initialize_coll_lbd_cpu(this, dcomm_handler, mesh)
        class(op_coll_lbd_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: lb_stripped(5), ub_stripped(5)
        ! Lower and upper bounds of the domain
        integer :: n, n_sp

        ! TODO: Update flops and ls
        this%n_flops = 75   ! We estimate division with 4
        this%n_ls    = 10

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        this%n_vp = this%mesh%size_vp()
        this%n_mu = this%mesh%size_mu()

        ! Prefactors needed for numerics
        this%delta_vpar     = this%mesh%delta_vp()
        this%delta_sqrt_mu  = this%mesh%delta_sqrt_mu()
        this%inv_delta_vpar = 1.0_GP / this%delta_vpar

        ! Check relaxation type
        if(get_relx_type() /= "et" .and. get_relx_type() /= "em") then
            call handle_error("Selected relaxation type "//get_relx_type() &
                              //" is not supported for LBD collisions!", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        ! Check vp quadrature.
        ! We require midpoint for the current implementation
        if(this%mesh%quad_type_vp() /= "midpoint") then
            call handle_error("LBD collisions require midpoint quadrature &
                              &in vp! (was "//this%mesh%quad_type_vp()//")", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        ! Check mu spacing. We require quadratic for the current implementation
        ! so the sqrt mu grid is uniform.
        if(this%mesh%grid_type_mu() /= "quadratic") then
            call handle_error("LBD collisions require quadratic spacing &
                              &in mu! (was "//this%mesh%grid_type_mu()//")", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        this%prefac_nu = get_coll_ref()
        n_sp = this%mesh%size_sp()

        ! Calculate prefactor for B parallel star
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

        ! Allocate buffer for boundary sums
        allocate(this%da_boundary_sums)
        call this%da_boundary_sums%initialize( &
                                [lb_stripped(1), lb_stripped(2), 1, 1], &
                                [ub_stripped(1), ub_stripped(2), 4, n_sp])
        ! Allocate buffer for corrected moments
        allocate(this%da_corr_moments)
        call this%da_corr_moments%initialize( &
                                [lb_stripped(1), lb_stripped(2), 1, 1, 1], &
                                [ub_stripped(1), ub_stripped(2), 2, n_sp, n_sp])

    end subroutine

    module subroutine apply_coll_lbd_cpu(this, da_f_in, da_moments, da_f_out)
        class(op_coll_lbd_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: alpha, beta, m, l, k, i
        ! Loop indices for species loops alpha, beta
        ! and loop indices over real and velocity space m, l, k, i
        integer :: n_sp, lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        ! Total number of species, lower and upper bounds of the domain
        integer :: ierr
        ! Error flag
        real(kind=GP) :: m_alpha, m_beta, nufac
        ! Prefactors involving masses
        real(kind=GP) :: temp_scaling_alpha, temp_scaling_beta
        ! Prefactors involving temp scalings
        real(kind=GP) :: nu, u_ab, T_ab, sqrt_absB
        ! Intermediate quantities (prefactors for derivatives)
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in, f_out
        ! Pointer to the input and output distributions
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        ! Pointer to the output moments
        real(kind=GP), contiguous, pointer, dimension(:,:) :: collog
        ! Pointer to the Coulomb logarithm
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: corr_moments
        ! Pointer to the corrected moments
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, sqrt_mu
        ! Pointer to vp and sqrt_mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, &
                                                              curl_normb_y
        ! Pointer to absB and curl_normb_y
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        ! Pointer to compute point mask
        real(kind=GP) :: vpar_stag_p, vpar_stag_m, vperp_stag_p, vperp_stag_m
        ! Staggered parallel and perpendicular. Subscripts p/m indicate
        ! staggering in positive or negative direction, i.e. index +- half.
        real(kind=GP) :: vperp_0, delta_vperp, inv_delta_vperp
        ! Perpendicular velocity at grid point (0) and delta vperp
        real(kind=GP) :: bps_0, bps_p, bps_m
        ! Bps at grid point (0) and at neighbouring vpar points
        real(kind=GP) :: f_0, f_vpar_p, f_vpar_m, f_vperp_p, f_vperp_m
        ! Values of the distribution function at the grid point (0) and
        ! at neighbouring vpar and vperp points.
        real(kind=GP) :: fBps_vpar_stag_p, fBps_vpar_stag_m, &
                         f_vperp_stag_p, f_vperp_stag_m
        ! Values of f staggered in vperp direction and f*Bps staggered in vpar
        real(kind=GP) :: dfBps_dvpar_stag_p, dfBps_dvpar_stag_m, &
                         df_dvperp_stag_p, df_dvperp_stag_m
        ! Derivatives of staggered f and staggered f*Bps w.r.t. vpar and vperp
        real(kind=GP) :: F_par_conv_p, F_par_conv_m, &
                         F_par_diff_p, F_par_diff_m, &
                         F_perp_conv_p, F_perp_conv_m, &
                         F_perp_diff_p, F_perp_diff_m, F_par, F_perp
        ! Fluxes: parallel and perpendicular, convective and diffusive,
        !         positive and negative

        ! This subroutine applies the operator to the distribution function. It
        ! calculates the collision frequency of species alpha with species beta
        ! and the corresponding collision operator for alpha-beta collisions.

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
        corr_moments => this%da_corr_moments%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped() * int(n_sp, kind=INT64)
        call this%perf_counter%start_measurement()

        vp           => this%mesh%get_vp_pointer()
        sqrt_mu      => this%mesh%get_sqrt_mu_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()
        absB         => this%mesh%get_absB_pointer()
        is_compute   => this%mesh%get_is_compute_pointer()

        ! Calculate corrected moments to achieve discrete conservation
        ! of the operator.
        call profiler_start("calc_boundary_terms", ierr, set_path=.true.)
        call this%calc_boundary_terms(da_f_in, this%da_boundary_sums)
        call profiler_stop("calc_boundary_terms", ierr, set_path=.true.)

        call profiler_start("calc_corrected_moments", ierr)
        call this%calc_corrected_moments(da_moments, this%da_boundary_sums, &
                                         this%da_corr_moments)
        call profiler_stop("calc_corrected_moments", ierr)

        call profiler_start("compute", ierr)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, n_sp) &
        !$omp shared(this, f_in, f_out, moments, collog, corr_moments, &
        !$omp        sqrt_mu, vp, absB, is_compute, curl_normb_y) &
        !$omp private(i, k, l, m, alpha, beta, m_alpha, m_beta, nufac, &
        !$omp         temp_scaling_alpha, temp_scaling_beta, &
        !$omp         nu, u_ab, T_ab, bps_0, bps_p, bps_m, sqrt_absB, &
        !$omp         vperp_0, delta_vperp, inv_delta_vperp, &
        !$omp         vpar_stag_p, vpar_stag_m, vperp_stag_p, vperp_stag_m, &
        !$omp         f_0, f_vpar_p, f_vpar_m, f_vperp_p, f_vperp_m, &
        !$omp         fBps_vpar_stag_p, fBps_vpar_stag_m, &
        !$omp         f_vperp_stag_p, f_vperp_stag_m, dfBps_dvpar_stag_p, &
        !$omp         dfBps_dvpar_stag_m, df_dvperp_stag_p, df_dvperp_stag_m, &
        !$omp         F_par_conv_p, F_par_conv_m, F_par_diff_p, &
        !$omp         F_par_diff_m, F_perp_conv_p, F_perp_conv_m, &
        !$omp         F_perp_diff_p, F_perp_diff_m, F_par, F_perp)
        ! Loop over all species combinations
        do alpha = lb_stripped(5), ub_stripped(5)
        ! This has to go explicitly through all species because of mpi
        ! processes may be parallized on the species dimension
        do beta = 1, n_sp

            ! Masses for inner loops
            m_alpha = get_mass(alpha)
            m_beta  = get_mass(beta)
            temp_scaling_alpha = get_temp_scaling(alpha)
            temp_scaling_beta = get_temp_scaling(beta)

            ! Prefac nu with masses and charges
            nufac = this%prefac_nu * sqrt(m_alpha * m_beta) * &
                (get_charge(alpha) * get_charge(beta))**2

            ! Depending on relaxation type we adjust the prefactor of the
            ! collision frequency (it only changes for em). Other adjustments
            ! are done in the corrected moments directly.
            select case(get_relx_type())
                ! Temperature relaxation
                case("et")
                    nufac = nufac * 0.5_GP
                ! Momentum relaxation
                case("em")
                    nufac = nufac * 0.5_GP * (m_alpha + m_beta) / m_alpha
            end select

            do m = lb_stripped(4), ub_stripped(4)
            do l = lb_stripped(3), ub_stripped(3)
            do k = lb_stripped(2), ub_stripped(2)
            !$omp do schedule(static)
            do i = lb_stripped(1), ub_stripped(1)

                ! Mixed flow and temperature for alpha-beta collision
                u_ab = corr_moments(i, k, 1, alpha, beta)
                T_ab = corr_moments(i, k, 2, alpha, beta)

                ! Collision frequency
                nu = nufac * collog(i, k) &
                   * moments(i, k, 1, beta) * &
                    (m_alpha * moments(i, k, 3, beta) * temp_scaling_beta + &
                     m_beta  * moments(i, k, 3, alpha) * temp_scaling_alpha &
                    )**(-1.5_GP)

                ! Precalculation of sqrt B and perp velocities
                sqrt_absB       = sqrt(absB(i, k))
                delta_vperp     = this%delta_sqrt_mu * sqrt_absB
                inv_delta_vperp = 1.0_GP / delta_vperp

                ! Staggered vp and mu
                vpar_stag_p = vp(l) + 0.5_GP * this%delta_vpar
                vpar_stag_m = vp(l) - 0.5_GP * this%delta_vpar

                vperp_0 = sqrt_mu(m) * sqrt_absB
                vperp_stag_p = vperp_0 + 0.5_GP * delta_vperp
                vperp_stag_m = vperp_0 - 0.5_GP * delta_vperp

                ! Get values of f (like 2D, 5 point stencil)
                f_0       = f_in(i, k, l    , m    , alpha)
                f_vpar_p  = f_in(i, k, l + 1, m    , alpha)
                f_vpar_m  = f_in(i, k, l - 1, m    , alpha)
                f_vperp_p = f_in(i, k, l    , m + 1, alpha)
                f_vperp_m = f_in(i, k, l    , m - 1, alpha)

                ! Calculate B parallel star
                bps_0 = absB(i, k) + this%prefac_bps(alpha) &
                                   * vp(l) * curl_normb_y(i, k)

                ! NOTE: We set the values of bps_m and bps_p to zero at
                !       the boundaries because it is not used due to the
                !       boundary conditions and we want to avoid undefined
                !       behaviour.

                ! Calculate B parallel star at left vp coordinate
                if(l /= 1) then
                    bps_m = absB(i, k) + this%prefac_bps(alpha) &
                                       * vp(l - 1) * curl_normb_y(i, k)
                else
                    bps_m = 0.0_GP
                end if
                ! Calculate B parallel star at right vp coordinate
                if(l /= this%n_vp) then
                    bps_p = absB(i, k) + this%prefac_bps(alpha) &
                                       * vp(l + 1) * curl_normb_y(i, k)
                else
                    bps_p = 0.0_GP
                end if

                ! Staggered f*Bps and f
                fBps_vpar_stag_p = 0.5_GP * (f_vpar_p * bps_p + f_0 * bps_0)
                fBps_vpar_stag_m = 0.5_GP * (f_vpar_m * bps_m + f_0 * bps_0)
                f_vperp_stag_p   = 0.5_GP * (f_vperp_p + f_0)
                f_vperp_stag_m   = 0.5_GP * (f_vperp_m + f_0)

                ! Derivatives of f*Bps and f
                dfBps_dvpar_stag_p = this%inv_delta_vpar &
                                   * (f_vpar_p * bps_p - f_0 * bps_0)
                dfBps_dvpar_stag_m = this%inv_delta_vpar &
                                   * (f_0 * bps_0 - f_vpar_m * bps_m)
                df_dvperp_stag_p = inv_delta_vperp * (f_vperp_p - f_0)
                df_dvperp_stag_m = inv_delta_vperp * (f_0 - f_vperp_m)

                ! Convective and diffusive parallel flux
                F_par_conv_p =  (vpar_stag_p - u_ab) * fBps_vpar_stag_p
                F_par_conv_m = -(vpar_stag_m - u_ab) * fBps_vpar_stag_m
                F_par_diff_p =  0.5_GP * T_ab * dfBps_dvpar_stag_p
                F_par_diff_m = -0.5_GP * T_ab * dfBps_dvpar_stag_m

                ! Convective and diffusive perpendicular flux
                F_perp_conv_p =  vperp_stag_p**2 * bps_0 &
                                        * f_vperp_stag_p / vperp_0
                F_perp_conv_m = -vperp_stag_m**2 * bps_0 &
                                        * f_vperp_stag_m / vperp_0
                F_perp_diff_p =  0.5_GP * T_ab * vperp_stag_p &
                                        * bps_0 * df_dvperp_stag_p / vperp_0
                F_perp_diff_m = -0.5_GP * T_ab * vperp_stag_m &
                                        * bps_0 * df_dvperp_stag_m / vperp_0

                ! Apply boundary conditions
                if(m == 1) then
                    ! Negative perpendicular flux zero at lower mu bound
                    F_perp_conv_m = 0.0_GP
                    F_perp_diff_m = 0.0_GP
                end if
                if(m == this%n_mu) then
                    ! Positive perpendicular flux zero at upper mu bound
                    F_perp_conv_p = 0.0_GP
                    F_perp_diff_p = 0.0_GP
                end if
                if(l == 1) then
                    ! Negative parallel flux zero at lower vp bound
                    F_par_conv_m = 0.0_GP
                    F_par_diff_m = 0.0_GP
                end if
                if(l == this%n_vp) then
                    ! Positive parallel flux zero at upper vp bound
                    F_par_conv_p = 0.0_GP
                    F_par_diff_p = 0.0_GP
                end if

                ! Collect terms for the total fluxes
                F_par  = F_par_conv_p + F_par_diff_p &
                       + F_par_conv_m + F_par_diff_m
                F_perp = F_perp_conv_p + F_perp_diff_p &
                       + F_perp_conv_m + F_perp_diff_m

                ! Collect all pieces of the collision operator
                f_out(i, k, l, m, alpha) = f_out(i, k, l, m, alpha) &
                        + (nu / bps_0) * ( this%inv_delta_vpar * F_par &
                                         + inv_delta_vperp * F_perp)

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

        call profiler_stop("compute", ierr)

        call this%perf_counter%end_measurement()

    end subroutine

    module subroutine calc_boundary_terms(this, da_f_in, da_boundary_sums)
        class(op_coll_lbd_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_boundary_sums

        integer :: i, k, l, m, n, ierr
        ! Loop indices
        integer :: n_sp, lb_stripped(5), ub_stripped(5)
        ! Total number of species, lower and upper bounds of the domain
        real(kind=GP) :: prefac_sum, bps
        ! Prefactor for the boundary sums, B parallel star
        real(kind=GP) :: b_sum
        ! Temporary variable to store boundary sum
        real(kind=GP) :: sqrt_absB, inv_sqrt_absB, inv_absB
        ! Buffers for precalculated absB expressions
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in
        ! Pointer to the input distributions
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: boundary_sums
        ! Pointer to the boundary terms
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, sqrt_mu
        ! Pointer to vp and sqrt_mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, &
                                                              curl_normb_y
        ! Pointer to absB and curl_normb_y
        real(kind=GP), allocatable, dimension(:,:,:,:) :: send_buffer
        ! Buffer for boundary sums of individual MPI processes

        n_sp = this%mesh%size_sp()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer()
        boundary_sums => da_boundary_sums%get_pointer()

        vp            => this%mesh%get_vp_pointer()
        sqrt_mu       => this%mesh%get_sqrt_mu_pointer()
        absB          => this%mesh%get_absB_pointer()
        curl_normb_y  => this%mesh%get_curl_normb_y_pointer()

        ! Normalized velocity space volume element (absB multiplied later)
        prefac_sum = 0.5_GP * pi * this%delta_vpar * this%delta_sqrt_mu

        ! Allocate send_buffer
        if(.not. allocated(send_buffer)) then
            allocate(send_buffer(lb_stripped(1):ub_stripped(1), &
                                 lb_stripped(2):ub_stripped(2), 4, n_sp))
        end if

        ! First touch init of send_buffer
        call this%op_set_uniform%apply(send_buffer, 0.0_GP)

        call profiler_start("reduction", ierr)

        ! Calculate the velocity space boundary sums needed for the
        ! corrected moments.
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, f_in, vp, sqrt_mu, absB, send_buffer, &
        !$omp        curl_normb_y, prefac_sum) &
        !$omp private(i, k, l, m, n, bps, b_sum, &
        !$omp         sqrt_absB, inv_sqrt_absB, inv_absB)
        do n = lb_stripped(5), ub_stripped(5)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            sqrt_absB     = sqrt(absB(i, k))
            inv_sqrt_absB = 1.0_GP / sqrt_absB
            inv_absB      = 1.0_GP / absB(i, k)

            ! NOTE: The following if-statements will only allow processes
            !       that contain the corresponding part of f_in to do the
            !       sum over the vspace boundaries.

            ! Lower bound of vp
            if(lb_stripped(3) == 1) then
                b_sum = 0.0_GP
                ! Bps at lower vp boundary
                bps = absB(i, k) &
                    + this%prefac_bps(n) * vp(1) * curl_normb_y(i, k)
                ! Sum over mu line at lower vp boundary
                do m = lb_stripped(4), ub_stripped(4)
                    b_sum = b_sum &
                          + f_in(i, k, 1, m, n) * bps &
                            * sqrt_mu(m) * inv_sqrt_absB
                end do
                send_buffer(i, k, 1, n) = b_sum * prefac_sum * sqrt_absB
            end if

            ! Upper bound of vp
            if(ub_stripped(3) == this%n_vp) then
                b_sum = 0.0_GP
                ! Bps at upper vp boundary
                bps = absB(i, k) &
                    + this%prefac_bps(n) * vp(this%n_vp) * curl_normb_y(i, k)
                ! Sum over mu line at upper vp boundary
                do m = lb_stripped(4), ub_stripped(4)
                    b_sum = b_sum &
                          + f_in(i, k, this%n_vp, m, n) * bps &
                            * sqrt_mu(m) * inv_sqrt_absB
                end do
                send_buffer(i, k, 2, n) = b_sum * prefac_sum * sqrt_absB
            end if

            ! Lower bound of mu
            if(lb_stripped(4) == 1) then
                b_sum = 0.0_GP
                ! Sum over vp line at lower mu boundary
                do l = lb_stripped(3), ub_stripped(3)
                    bps = absB(i, k) &
                        + this%prefac_bps(n) * vp(l) * curl_normb_y(i, k)
                    b_sum = b_sum + f_in(i, k, l, 1, n) * bps * inv_absB
                end do
                send_buffer(i, k, 3, n) = b_sum * prefac_sum * sqrt_absB
            end if

            ! Upper bound of mu
            if(ub_stripped(4) == this%n_mu) then
                b_sum = 0.0_GP
                ! Sum over vp line at upper mu boundary
                do l = lb_stripped(3), ub_stripped(3)
                    bps = absB(i, k) &
                        + this%prefac_bps(n) * vp(l) * curl_normb_y(i, k)
                    b_sum = b_sum &
                          + f_in(i, k, l, this%n_mu, n) * bps * inv_absB
                end do
                send_buffer(i, k, 4, n) = b_sum * prefac_sum * sqrt_absB
            end if

        end do
        !$omp end do nowait
        end do
        end do
        !$omp end parallel

        call profiler_stop("reduction", ierr)

        ! Collect results of the boundary sums from all mpi processes
        call profiler_start_allreduce(this%dcomm_handler%get_comm_vp_mu_sp(), &
                                      ierr)

        call MPI_Allreduce(send_buffer, boundary_sums, &
                           size(send_buffer), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        call profiler_stop_allreduce(ierr)
    end subroutine

    module subroutine calc_corrected_moments(this, da_moments, &
                                             da_boundary_sums, da_corr_moments)
        class(op_coll_lbd_cpu_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_moments
        class(data_array_4d_t), intent(in) :: da_boundary_sums
        class(data_array_5d_t), intent(inout) :: da_corr_moments

        integer :: i, k, alpha, beta
        ! Loop indices
        integer :: n_sp, lb_stripped(2), ub_stripped(2)
        ! Total number of species, lower and upper bounds of the domain
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        ! Pointer to the input moments
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: boundary_sums
        ! Pointer to the boundary terms
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: corr_moments
        ! Pointer to the corrected moments
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, sqrt_mu
        ! Pointer to vp and sqrt_mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB
        ! Pointer to absB
        real(kind=GP) :: prefac1a, prefac2a, prefac3a, prefac4a, prefac5a, &
                         prefac6a, prefac1b, prefac2b, prefac3b, prefac4b, &
                         prefac5b, prefac6b
        ! Prefactors of boundary correction terms for species alpha, beta
        real(kind=GP) :: S1a, S2a, S3a, S4a, S1b, S2b, S3b, S4b
        ! Boundary sums for species alpha, beta
        real(kind=GP) :: sysfac1, sysfac2, sysfac3, sysfac4, sysfac5, sysfac6
        ! Prefactors of coupled system of equations for species alpha, beta
        real(kind=GP) :: delta, delta_T, gamma_a, gamma_b, epsilon_a, epsilon_b
        ! Prefactors for the system of equations coefficients
        real(kind=GP) :: inv_M0_a, inv_M0_b, Mu_alpha, Mu_beta, &
                         MTL_alpha, MTL_beta
        ! Moments
        real(kind=GP) :: temp_floor
        ! Temperature floor value
        real(kind=GP) :: vpar_min_stag_m, vpar_max_stag_p, &
                         sqrt_mu_min_stag_m, sqrt_mu_max_stag_p
        ! Staggered maximum and minimum vp and sqrt_mu
        real(kind=GP) :: sqrt_absB, vperp_min_stag_m, vperp_max_stag_p, &
                         delta_vperp, inv_delta_vperp
        ! Buffers for sqrt of absB and perpendicular velocity factors
        real(kind=GP) :: determinant
        ! Determinant of linear system of equations
        integer :: ierr, buff_bounds(4)
        ! Error code and buffer for domain bounds

        n_sp = this%mesh%size_sp()
        buff_bounds = da_moments%get_lbound_stripped()
        lb_stripped = buff_bounds(1:2)
        buff_bounds = da_moments%get_ubound_stripped()
        ub_stripped = buff_bounds(1:2)

        moments => da_moments%get_readonly_pointer()
        boundary_sums => da_boundary_sums%get_readonly_pointer()
        corr_moments => da_corr_moments%get_pointer()

        vp      => this%mesh%get_vp_pointer()
        sqrt_mu => this%mesh%get_sqrt_mu_pointer()
        absB    => this%mesh%get_absB_pointer()

        ! Staggered vp and sqrt_mu values at the boundaries
        vpar_min_stag_m    = vp(1) - 0.5_GP * this%delta_vpar
        vpar_max_stag_p    = vp(this%n_vp) + 0.5_GP * this%delta_vpar
        sqrt_mu_min_stag_m = sqrt_mu(1) - 0.5_GP * this%delta_sqrt_mu
        sqrt_mu_max_stag_p = sqrt_mu(this%n_mu) + 0.5_GP * this%delta_sqrt_mu

        ! Temperature floor in case the corrected temperature becomes
        ! too small or negative. We save it here to avoid a function
        ! call in the loop.
        temp_floor = get_temp_floor()
        ierr = GENEX_SUCCESS

        ! Calculate the corrected moments
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, n_sp) &
        !$omp shared(this, moments, absB, boundary_sums, corr_moments, &
        !$omp        ierr, vpar_min_stag_m, vpar_max_stag_p, &
        !$omp        sqrt_mu_min_stag_m, sqrt_mu_max_stag_p, temp_floor) &
        !$omp private(i, k, alpha, beta, inv_M0_a, inv_M0_b, &
        !$omp         delta, delta_T, gamma_a, gamma_b, epsilon_a, epsilon_b, &
        !$omp         sqrt_absB, vperp_min_stag_m, vperp_max_stag_p, &
        !$omp         delta_vperp, inv_delta_vperp, &
        !$omp         prefac1a, prefac2a, prefac3a, prefac4a, prefac5a, &
        !$omp         prefac6a, prefac1b, prefac2b, prefac3b, prefac4b, &
        !$omp         prefac5b, prefac6b, &
        !$omp         S1a, S2a, S3a, S4a, S1b, S2b, S3b, S4b, &
        !$omp         sysfac1, sysfac2, sysfac3, sysfac4, sysfac5, sysfac6, &
        !$omp         determinant, Mu_alpha, Mu_beta, MTL_alpha, MTL_beta)
        do alpha = 1, n_sp
        do beta = 1, n_sp

            ! Depending on relaxation type we adjust the prefactors for the
            ! system of equations. We do this before the main loop to
            ! avoid conditionals and to simplify code.
            delta = sqrt((get_temp_scaling(alpha) / get_mass(alpha)) &
                        /(get_temp_scaling(beta) / get_mass(beta)))
            delta_T = get_temp_scaling(alpha) / get_temp_scaling(beta)
            select case(get_relx_type())
                ! Temperature relaxation
                case("et")
                    gamma_a = sqrt(get_mass(alpha) * get_temp_scaling(alpha))
                    gamma_b = sqrt(get_mass(beta) * get_temp_scaling(alpha))
                    epsilon_a = get_temp_scaling(alpha)
                    epsilon_b = get_temp_scaling(beta)
                ! Momentum relaxation
                case("em")
                    gamma_a = sqrt(get_temp_scaling(alpha) / get_mass(alpha))
                    gamma_b = sqrt(get_temp_scaling(beta) / get_mass(beta))
                    epsilon_a = get_temp_scaling(alpha) / get_mass(alpha)
                    epsilon_b = get_temp_scaling(beta) / get_mass(beta)
            end select

        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            ! Get boundary sums
            S1a = boundary_sums(i, k, 1, alpha)
            S2a = boundary_sums(i, k, 2, alpha)
            S3a = boundary_sums(i, k, 3, alpha)
            S4a = boundary_sums(i, k, 4, alpha)
            S1b = boundary_sums(i, k, 1, beta)
            S2b = boundary_sums(i, k, 2, beta)
            S3b = boundary_sums(i, k, 3, beta)
            S4b = boundary_sums(i, k, 4, beta)

            ! Inverse 0th moments
            inv_M0_a  = 1.0_GP / moments(i, k, 1, alpha)
            inv_M0_b  = 1.0_GP / moments(i, k, 1, beta)
            ! 1st moments normalized by 0th moments
            ! (this is what the moment operator calculated in the 2nd entry)
            Mu_alpha  = moments(i, k, 2, alpha)
            Mu_beta   = moments(i, k, 2, beta)
            ! 2nd moments in lab frame (add Mu) normalized by 0th moments
            MTL_alpha = 1.5_GP * moments(i, k, 3, alpha) + Mu_alpha**2
            MTL_beta  = 1.5_GP * moments(i, k, 3, beta) + Mu_beta**2

            sqrt_absB        = sqrt(absB(i, k))
            vperp_min_stag_m = sqrt_mu_min_stag_m * sqrt_absB
            vperp_max_stag_p = sqrt_mu_max_stag_p * sqrt_absB
            delta_vperp      = this%delta_sqrt_mu * sqrt_absB
            inv_delta_vperp  = 1.0_GP / delta_vperp

            ! Calculate the prefactors from the boundary correction terms

            ! Terms of PBC - species alpha
            prefac1a = -2.0_GP * (S1a + S2a)
            prefac2a =  2.0_GP * (S1a - S2a) * this%inv_delta_vpar
            prefac3a =  2.0_GP * (vpar_min_stag_m * S1a + vpar_max_stag_p * S2a)
            ! Terms of PBC - species beta
            prefac1b = -2.0_GP * (S1b + S2b)
            prefac2b =  2.0_GP * (S1b - S2b) * this%inv_delta_vpar
            prefac3b =  2.0_GP * (vpar_min_stag_m * S1b + vpar_max_stag_p * S2b)
            ! Terms of EBC - species alpha
            prefac4a = -0.5_GP * prefac3a
            prefac5a = inv_delta_vperp * ( vperp_min_stag_m**2 * S3a &
                                         - vperp_max_stag_p**2 * S4a) &
                     + this%inv_delta_vpar * ( vpar_min_stag_m * S1a &
                                             - vpar_max_stag_p * S2a)
            prefac6a = (vperp_min_stag_m**3 * S3a + vperp_max_stag_p**3 * S4a) &
                     + vpar_min_stag_m**2 * S1a + vpar_max_stag_p**2 * S2a
            ! Terms of EBC - species beta
            prefac4b = -0.5_GP * prefac3b
            prefac5b = inv_delta_vperp * ( vperp_min_stag_m**2 * S3b &
                                         - vperp_max_stag_p**2 * S4b) &
                     + this%inv_delta_vpar * ( vpar_min_stag_m * S1b &
                                             - vpar_max_stag_p * S2b)
            prefac6b = (vperp_min_stag_m**3* S3b + vperp_max_stag_p**3 * S4b) &
                     + vpar_min_stag_m**2 * S1b + vpar_max_stag_p**2 * S2b

            ! Solve the system of equations for the corrected moments

            ! Calculate the factors for the system of equations
            sysfac1 = gamma_a &
                    + gamma_b * delta &
                    + gamma_a * inv_M0_a * prefac1a &
                    + gamma_b * inv_M0_b * prefac1b * delta
            sysfac2 = gamma_a * inv_M0_a * prefac2a &
                    + gamma_b * inv_M0_b * prefac2b * delta_T
            sysfac3 = gamma_a * Mu_alpha &
                    + gamma_b * Mu_beta &
                    - gamma_a * inv_M0_a * prefac3a &
                    - gamma_b * inv_M0_b * prefac3b
            sysfac4 = epsilon_a * Mu_alpha &
                    + epsilon_b * Mu_beta * delta &
                    + 2.0_GP * epsilon_a * inv_M0_a * prefac4a &
                    + 2.0_GP * epsilon_b * inv_M0_b * prefac4b * delta
            sysfac5 = 1.5_GP * (epsilon_a + epsilon_b * delta_T) &
                    + 2.0_GP * epsilon_a * inv_M0_a * prefac5a &
                    + 2.0_GP * epsilon_b * inv_M0_b * prefac5b * delta_T
            sysfac6 = epsilon_a * MTL_alpha &
                    + epsilon_b * MTL_beta &
                    + ( 0.25_GP * this%delta_vpar**2 &
                      + 0.75_GP * delta_vperp**2) * (epsilon_a + epsilon_b) &
                    - 2.0_GP * epsilon_a * inv_M0_a * prefac6a &
                    - 2.0_GP * epsilon_b * inv_M0_b * prefac6b

            ! NOTE: First solving for T and then for u is more robust
            !       since sysfac2 may become zero. However in some corner cases
            !       the denominator (determinant of the system) may become
            !       zero. Thus we check and avoid the divide by zero.

            ! Calculate determinant and check if it is nonzero
            determinant = sysfac1 * sysfac5 - sysfac4 * sysfac2

            if(determinant < 10.0_GP * GP_EPS) then
                ! NOTE: This case may happen for rare corner cases such as for
                !       constant distribution function in the EM version.

                ! Calculate sysfac without boundary corrections. This should
                ! not lead to zero determinant.
                sysfac1 = gamma_a + gamma_b * delta
                sysfac2 = 0.0_GP
                sysfac3 = gamma_a * Mu_alpha + gamma_b * Mu_beta
                sysfac4 = epsilon_a * Mu_alpha + epsilon_b * Mu_beta * delta
                sysfac5 = 1.5_GP * (epsilon_a + epsilon_b * delta_T)
                sysfac6 = epsilon_a * MTL_alpha + epsilon_b * MTL_beta &
                        + ( 0.25_GP * this%delta_vpar**2 &
                          + 0.75_GP * delta_vperp**2) * (epsilon_a + epsilon_b)
                determinant = sysfac1 * sysfac5 - sysfac4 * sysfac2

                ierr = GENEX_ERR_COLL
                cycle
            endif

            ! First solve for corrected temperature T (index 2)
            corr_moments(i, k, 2, alpha, beta) &
                    = (sysfac1 * sysfac6 - sysfac4 * sysfac3) / determinant

            ! Second solve for corrected flow u (index 1)
            corr_moments(i, k, 1, alpha, beta) &
                    = (sysfac3 - sysfac2 &
                        * corr_moments(i, k, 2, alpha, beta)) / sysfac1

            ! Avoid negative temperatures with temp floor
            corr_moments(i, k, 2, alpha, beta) = &
                max(corr_moments(i, k, 2, alpha, beta), temp_floor)

        end do
        !$omp end do nowait
        end do
        end do
        end do
        !$omp end parallel

        ! NOTE: We do not consider the case of zero determinant in the linear
        !       system as fatal because we only apply the grid dependent
        !       corrections then. We print a warning.
        if(ierr /= GENEX_SUCCESS) then
            call handle_error("Determinant of the linear system in corrected &
                              &moments was zero (in LBD collisions)!", &
                              GENEX_WRN_COLL, __LINE__, __FILE__)
        endif

    end subroutine

end submodule
