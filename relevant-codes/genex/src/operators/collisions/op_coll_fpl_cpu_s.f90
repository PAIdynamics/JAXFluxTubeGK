submodule(op_coll_m) op_coll_fpl_cpu_s
    !! Contains the implementation of op_coll_fpl_cpu_t,
    !! the Fokker-Planck-Landau collision operator.

    use MPI
    use, intrinsic :: iso_fortran_env
    use math_m, only: pi
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_WRN_COLL, &
                                    GENEX_ERR_COLL
    use genex_fortran_env_m, only: MPI_GP, GP_EPS, DP
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_normalization_m, only: get_coll_ref
    implicit none

    real(kind=GP), parameter :: zero_tol = 100.0_GP * GP_EPS
    !! Tolerance for comparison with zero
contains

    module subroutine initialize_coll_fpl_cpu(this, dcomm_handler, mesh)
        class(op_coll_fpl_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: l, m, n
        ! Loop indices
        integer :: lb_stripped(5), ub_stripped(5)
        ! Lower and upper bounds of the domain without ghost cells
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, sqrt_mu
        ! Pointer to vp and sqrt_mu of the mesh

        ! TODO: Estimate flops and ls
        this%n_flops = 1
        this%n_ls    = 1

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

        this%delta_vp      = this%mesh%delta_vp()
        this%delta_sqrt_mu = this%mesh%delta_sqrt_mu()

        this%n_sp      = this%mesh%size_sp()
        this%n_vp      = this%mesh%size_vp()
        this%n_mu      = this%mesh%size_mu()
        this%n_vp_stag = this%n_vp - 1
        this%n_mu_stag = this%n_mu - 1

        ! Check vp quadrature.
        ! We require midpoint for the current implementation
        if(this%mesh%quad_type_vp() /= "midpoint") then
            call handle_error("FPL collisions require midpoint quadrature &
                              &in vp! (was "//this%mesh%quad_type_vp()//")", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        ! Check mu spacing. We require quadratic for the current implementation
        ! so the sqrt mu grid is uniform.
        if(this%mesh%grid_type_mu() /= "quadratic") then
            call handle_error("FPL collisions require quadratic spacing &
                              &in mu! (was "//this%mesh%grid_type_mu()//")", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        end if

        vp      => this%mesh%get_vp_pointer()
        sqrt_mu => this%mesh%get_sqrt_mu_pointer()
        ! Create staggered grids
        ! NOTE: Index of staggered grid point corresponds to usual grid point
        !       as: i (staggered) -> i+1/2 (usual)
        allocate(this%vp_stag(0:this%n_vp))
        do l = 1, this%n_vp_stag
            this%vp_stag(l) = vp(l) + 0.5_GP * this%delta_vp
        enddo
        ! We need to extend the staggered grid up to 1 point in each direction
        ! (used as ghost points).
        this%vp_stag(0)         = this%vp_stag(1) - this%delta_vp
        this%vp_stag(this%n_vp) = this%vp_stag(this%n_vp - 1) + this%delta_vp

        allocate(this%sqrt_mu_stag(0:this%n_mu))
        do m = 1, this%n_mu_stag
            this%sqrt_mu_stag(m) = sqrt_mu(m) + 0.5_GP * this%delta_sqrt_mu
        enddo
        this%sqrt_mu_stag(0)         = this%sqrt_mu_stag(1) - this%delta_sqrt_mu
        this%sqrt_mu_stag(this%n_mu) = this%sqrt_mu_stag(this%n_mu - 1) &
                                     + this%delta_sqrt_mu

        allocate(this%prefac_bps(this%n_sp))
        do n = 1, this%n_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
        enddo

        allocate(this%da_collog)
        lb_stripped = dcomm_handler%get_lbound_stripped()
        ub_stripped = dcomm_handler%get_ubound_stripped()
        call this%da_collog%initialize(lb_stripped(1:2), ub_stripped(1:2))
        call this%initialize_staggered_bounds(lb_stripped, ub_stripped)
    end subroutine

    module subroutine apply_coll_fpl_cpu(this, da_f_in, da_moments, da_f_out)
        class(op_coll_fpl_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: i, k, l, m, n, lprime, mprime, nprime
        ! Loop variables
        integer :: lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        ! Lower and upper bounds of the domain
        integer :: ierr
        ! Error code
        logical :: err_interp, err_ceie, err_ceik
        ! Error flags for routines in loop
        real(kind=GP) :: inv_mass_n, inv_sqrt_masses
        ! Inverse of mass(n) and inverse of the square root of the mass product
        real(kind=GP) :: temp_scaling_n, temp_scaling_nprime
        ! Temp scalings
        real(kind=GP) :: inv_temp_scaling_n, inv_sqrt_temp_scalings
        ! Inverse of temp_scaling and square root of the product
        real(kind=GP) :: prefac_gamma
        ! Real space independent prefactor for gamma
        real(kind=GP) :: gamma_D, gamma_E
        ! Prefactor for E and D
        real(kind=GP) :: inv_dvpar, inv_dvperp, prefac_inv_dvperp
        ! Inverse perpendicular and parallel velocity
        real(kind=GP) :: sqrt_absB
        ! Precalculated square root of absB
        real(kind=GP) :: f00, f0p, fp0, fpp
        ! Distribution function on grid points
        real(kind=GP) :: finterp, fdiff_par, fdiff_perp
        ! Interpolated distribution function and derivatives
        real(kind=GP) :: delta_par_p, delta_par_m, delta_perp_p, delta_perp_m
        ! Interpolation weights for upper and lower points w.r.t. the actual
        ! grid point
        real(kind=GP) :: area, area_perp, area_par
        ! Full, perpendicular and parallel velocity space area element
        real(kind=GP) :: area_ratio_p, area_ratio_m
        ! Velocity space area element ratio in calculation of fluxes
        real(kind=GP) :: df_perp1, df_perp2, df_par1, df_par2
        ! Changes in distribution function due to collisional fluxes
        real(kind=GP), dimension(:,:,:,:,:), allocatable :: f_send, f_buffer
        ! Buffer for exchanged distribution function
        real(kind=GP) :: inv_temp
        ! Inverse mixing temperature used for interpolation weights
        real(kind=GP), dimension(1:this%n_vp) :: delta_par
        ! Parallel interpolation weights
        real(kind=GP), dimension(1:this%n_mu) :: delta_perp
        ! Perpendicular interpolation weights
        real(kind=GP) :: prefac_vprime, prefac_v
        ! Prefactor for (primed) velocities
        real(kind=GP) :: prefac_v_perpprime, prefac_v_perp
        ! Prefactor for (primed) perpendicular velocity
        real(kind=GP) :: v_par_diff, v_perpprime, v_perp
        ! Velocity space coordinates for velocity tensor evaluation
        real(kind=GP) :: lambda, karg, marg, prefac_U, sqrt_kp1
        ! Velocity tensor arguments
        real(kind=GP) :: Em, Km, I1, I2, I3
        ! Elliptic integrals and combinations of them
        real(kind=GP) :: U_papa, U_pepa, U_pepe, U_papep, U_pepep
        ! Velocity tensor at staggered grid points
        real(kind=GP) :: E_par, E_perp
        ! Parallel and perpendicular components of the friction vector
        real(kind=GP) :: D_par_par, D_perp_par, D_perp_perp
        ! Components of the diffusion tensor
        real(kind=GP), dimension(:,:), allocatable :: flux_perp, flux_par
        ! Perpendicular and parallel collisional flux
        ! NOTE: We define the fluxes on a larger grid to consider the
        !       boundary conditions. There we set the flux to zero.
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in, f_out
        ! Pointer to the input and output distributions
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        ! Pointer to the output moments
        real(kind=GP), contiguous, pointer, dimension(:,:) :: collog
        ! Pointer to the Coulomb logarithm
        real(kind=GP), contiguous, pointer, dimension(:) :: sqrt_mu, vp
        ! Pointer to sqrt mu and vp
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, &
                                                              curl_normb_y
        ! Pointer to absB and curl_normb_y
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        ! Pointer to compute point mask
        real(kind=GP) :: bps
        ! B parallel star

        lb = da_f_in%get_lbound()
        ub = da_f_in%get_ubound()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer()
        moments => da_moments%get_pointer()
        f_out => da_f_out%get_pointer()

        ! First time setup
        if(.not. this%is_initialized) then
            call calc_collog(da_moments, this%da_collog)
            this%is_initialized = .true.
        endif
        collog => this%da_collog%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped() &
                          * int(this%n_vp * this%n_mu * this%n_sp, kind=INT64)
        call this%perf_counter%start_measurement()

        sqrt_mu      => this%mesh%get_sqrt_mu_pointer()
        vp           => this%mesh%get_vp_pointer()
        absB         => this%mesh%get_absB_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()
        is_compute   => this%mesh%get_is_compute_pointer()

        allocate(flux_perp((lb_stripped(3)-1):ub_stripped(3), &
                           (lb_stripped(4)-1):ub_stripped(4)))
        allocate(flux_par, mold=flux_perp)

        ! Communicate vp, mu and sp dimension of f_in through f_send into
        ! f_buffer. The intermediate send buffer is needed because the source
        ! and the target of the communication have different shapes.
        allocate(f_buffer(lb_stripped(1):ub_stripped(1), &
                          lb_stripped(2):ub_stripped(2), &
                          1:this%n_vp, 1:this%n_mu, 1:this%n_sp))
        allocate(f_send, mold=f_buffer)
        ! NOTE: We use allreduce with summation for the communication and thus
        !       need to set all unused values of the send buffer to zero.
        call this%op_set_uniform%apply(f_send, 0.0_GP)

        call profiler_start("buffer_copy", ierr)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp private(i, k, l, m, n, bps) &
        !$omp shared(this, absB, vp, curl_normb_y, f_send, f_in)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            f_send(i, k, l, m, n) = f_in(i, k, l, m, n)
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("buffer_copy", ierr)

        call profiler_start_allreduce(this%dcomm_handler%get_comm_vp_mu_sp(), &
                                      ierr)

        call MPI_Allreduce(f_send, f_buffer, &
                           size(f_send), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        call profiler_stop_allreduce(ierr)

        inv_dvpar         = 1.0_GP / this%delta_vp
        prefac_inv_dvperp = 1.0_GP / this%delta_sqrt_mu

        call profiler_start("compute", ierr)

        ! NOTE: Error flags are omp shared since we only need to check if any
        !       task sets them to true or not.
        ! TODO: Investigate the failure of simd vectorization here due to
        !       the addition of fukushima_ceik_local and fukushima_ceie_local
        !       functions
        err_interp = .false.
        err_ceie   = .false.
        err_ceik   = .false.
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, inv_dvpar, &
        !$omp              prefac_inv_dvperp) &
        !$omp private(i, k, l, lprime, m, mprime, n, nprime, inv_mass_n, &
        !$omp         inv_sqrt_masses, prefac_gamma, gamma_D, gamma_E, &
        !$omp         temp_scaling_n, temp_scaling_nprime, &
        !$omp         inv_temp_scaling_n, inv_sqrt_temp_scalings, &
        !$omp         sqrt_absB, inv_dvperp, inv_temp, &
        !$omp         delta_par, delta_perp, flux_perp, flux_par, &
        !$omp         E_par, E_perp, D_par_par, D_perp_par, D_perp_perp, &
        !$omp         area_perp, area_par, area, area_ratio_p, area_ratio_m, &
        !$omp         f00, f0p, fp0, fpp, delta_par_p, delta_par_m, &
        !$omp         delta_perp_p, delta_perp_m, finterp, fdiff_par, &
        !$omp         fdiff_perp, U_papa, U_pepa, U_pepe, U_pepep, U_papep, &
        !$omp         df_perp1, df_perp2, df_par1, df_par2, bps, prefac_vprime,&
        !$omp         prefac_v, prefac_v_perpprime, prefac_v_perp, v_par_diff, &
        !$omp         v_perp, v_perpprime, lambda, sqrt_kp1, karg, marg, &
        !$omp         prefac_U, Km, Em, I1, I2, I3, ierr) &
        !$omp shared(this, absB, sqrt_mu, is_compute, moments, collog, &
        !$omp        f_buffer, f_out, vp, curl_normb_y, &
        !$omp        err_interp, err_ceie, err_ceik)
        do nprime = 1, this%n_sp
        do n = lb_stripped(5), ub_stripped(5)

            inv_mass_n      = 1.0_GP / get_mass(n)
            inv_sqrt_masses = 1.0_GP / sqrt(get_mass(n) * get_mass(nprime))
            temp_scaling_n      = get_temp_scaling(n)
            temp_scaling_nprime = get_temp_scaling(nprime)
            inv_temp_scaling_n = 1.0_GP / temp_scaling_n
            inv_sqrt_temp_scalings = 1.0_GP &
                                   / sqrt(temp_scaling_n * temp_scaling_nprime)
            prefac_gamma = get_coll_ref() * 3.0_GP / 16.0_GP * sqrt(pi) &
                         * (get_charge(n) * get_charge(nprime))**2
            prefac_vprime = sqrt(temp_scaling_nprime / get_mass(nprime))
            prefac_v = sqrt(temp_scaling_n / get_mass(n))

        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            gamma_D = -prefac_gamma * collog(i, k) &
                    * inv_mass_n * inv_temp_scaling_n
            gamma_E = prefac_gamma * collog(i, k) &
                    * inv_sqrt_temp_scalings * inv_sqrt_masses

            sqrt_absB  = sqrt(absB(i, k))
            inv_dvperp = prefac_inv_dvperp / sqrt_absB

            ! Calculate "inverse temperature" parameter used in interpolation
            ! weights. We avoid a zero divided by zero situation by setting
            ! inv_temp to one if the numerator and denominator are both zero.
            if((moments(i, k, 1, n)      < zero_tol) .and. &
               (moments(i, k, 1, nprime) < zero_tol)) then
                inv_temp = 1.0_GP
            else
                inv_temp = (moments(i, k, 1, n) + moments(i, k, 1, nprime)) &
                         / (moments(i, k, 1, n) &
                                * moments(i, k, 3, n) &
                                * temp_scaling_n &
                           + moments(i, k, 1, nprime) &
                                * moments(i, k, 3, nprime) &
                                * temp_scaling_nprime)
            endif
            ! Calculate interpolation weights for this real space point and
            ! species combination
            call this%calc_interp_weights(inv_temp, absB(i, k), &
                                          delta_par, delta_perp, ierr)
            if(ierr /= GENEX_SUCCESS) err_interp = .true.

            ! Set flux arrays to zero. Boundary values will not be overwritten
            ! but used in the stencil which enforces the boundary conditions.
            flux_perp = 0.0_GP
            flux_par  = 0.0_GP
            do m = this%lb_stagger(4), this%ub_stagger(4)
            do l = this%lb_stagger(3), this%ub_stagger(3)

                ! Calculate FPL coefficients by integrating over the primed
                ! primed velocity space

                E_par       = 0.0_GP
                E_perp      = 0.0_GP
                D_par_par   = 0.0_GP
                D_perp_par  = 0.0_GP
                D_perp_perp = 0.0_GP

                do lprime = 1, this%n_vp_stag
                do mprime = 1, this%n_mu_stag

                    prefac_v_perpprime = prefac_vprime &
                                       * this%sqrt_mu_stag(mprime)
                    prefac_v_perp = prefac_v * this%sqrt_mu_stag(m)
                    v_par_diff = prefac_v * this%vp_stag(l) &
                               - prefac_vprime * this%vp_stag(lprime)
                    ! Cycle when vpar and vperp are the same for primed and
                    ! unprimed index. In that case the elliptic integrals
                    ! are ill-defined.
                    if (abs(v_par_diff) < zero_tol .and. &
                        abs(prefac_v_perp - prefac_v_perpprime) < zero_tol) &
                        cycle

                    v_perp      = sqrt(absB(i, k)) * prefac_v_perp
                    v_perpprime = sqrt(absB(i, k)) * prefac_v_perpprime

                    lambda   = v_perp**2 + v_perpprime**2 + v_par_diff**2
                    karg     = 2.0_GP * v_perp * v_perpprime / lambda
                    sqrt_kp1 = sqrt(karg + 1.0_GP)
                    marg     = 2.0_GP * karg / (karg + 1.0_GP)
                    ! NOTE: Factor 4pi has been removed from the
                    !       definition of U.
                    prefac_U = lambda**(-1.5_GP)

                    ! Integrals from gyroaverage. Given as combinations of
                    ! elliptic integrals E(m) and K(m).
                    ! NOTE: In Yoon 2014 integrals are written as E(k), K(k)
                    !       but are meant as E(m) and K(m). In literature these
                    !       are different functions, connected via m=k**2.
                    ! NOTE: Using the local functions here allows the compiler
                    !       to vectorize which is important here since the
                    !       computations in this loop are expensive.
                    Km = fukushima_ceik_local(marg)
                    if(Km == 0.0_GP) err_ceik = .true.
                    Em = fukushima_ceie_local(marg)
                    if(Em == 0.0_GP) err_ceie = .true.

                    I1 = -4.0_GP / (karg**2 * sqrt_kp1) &
                       * ((karg + 1.0_GP) * Em - Km)
                    I2 = -2.0_GP / ((karg - 1.0_GP) * sqrt_kp1) * Em
                    I3 = -2.0_GP / ((karg - 1.0_GP) * sqrt_kp1 * karg) &
                       * (Em + (karg - 1.0_GP) * Km)

                    ! Components of velocity tensor, see appendix Yoon 2014
                    U_papa = prefac_U * ((v_perp**2 + v_perpprime**2) * I2 &
                                        - 2.0_GP * v_perp * v_perpprime * I3)
                    U_papep = prefac_U * v_par_diff &
                            * (v_perpprime * I2 - v_perp * I3)
                    U_pepa = prefac_U * v_par_diff &
                           * (v_perpprime * I3 - v_perp * I2)
                    U_pepe = prefac_U &
                           * (v_perpprime**2 * I1 + v_par_diff**2 * I2)
                    U_pepep = prefac_U &
                            * (v_par_diff**2 * I3 + v_perp * v_perpprime * I1)

                    area_perp = this%delta_sqrt_mu &
                              * this%sqrt_mu_stag(mprime) * sqrt_absB
                    area      = area_perp * this%delta_vp
                    area_par  = this%sqrt_mu_stag(mprime) * this%delta_vp

                    ! Apply B parallel star (=Jacobian) from primed
                    ! velocity space integration
                    bps = absB(i, k) + this%prefac_bps(nprime) &
                                     * this%vp_stag(lprime) * curl_normb_y(i, k)
                    area_perp = area_perp * bps
                    area      = area      * bps
                    area_par  = area_par  * bps

                    f00 = f_buffer(i, k, lprime    , mprime    , nprime)
                    f0p = f_buffer(i, k, lprime    , mprime + 1, nprime)
                    fp0 = f_buffer(i, k, lprime + 1, mprime    , nprime)
                    fpp = f_buffer(i, k, lprime + 1, mprime + 1, nprime)

                    delta_par_p  = delta_par(lprime)
                    delta_perp_p = delta_perp(mprime)
                    delta_par_m  = 1.0_GP - delta_par_p
                    delta_perp_m = 1.0_GP - delta_perp_p

                    finterp = delta_par_m * delta_perp_m * f00 &
                            + delta_par_m * delta_perp_p * f0p &
                            + delta_par_p * delta_perp_m * fp0 &
                            + delta_par_p * delta_perp_p * fpp

                    ! NOTE: Delta vpar and vperp included in area
                    fdiff_par  = delta_perp_p * fpp - delta_perp_p * f0p &
                               + delta_perp_m * fp0 - delta_perp_m * f00

                    fdiff_perp = delta_par_p * fpp - delta_par_p * fp0 &
                               + delta_par_m * f0p - delta_par_m * f00

                    D_par_par   = D_par_par   &
                                + gamma_D * area * U_papa * finterp
                    D_perp_par  = D_perp_par  &
                                + gamma_D * area * U_pepa * finterp
                    D_perp_perp = D_perp_perp &
                                + gamma_D * area * U_pepe * finterp

                    E_par  = E_par &
                           + gamma_E * area_par  * U_papep * fdiff_perp &
                           + gamma_E * area_perp * U_papa  * fdiff_par
                    E_perp = E_perp &
                           + gamma_E * area_par  * U_pepep * fdiff_perp &
                           + gamma_E * area_perp * U_pepa  * fdiff_par
                enddo
                enddo

                ! Apply B parallel star (=Jacobian) from phase space divergence
                ! (inner part)
                bps = absB(i, k) + this%prefac_bps(n) * this%vp_stag(l) &
                                 * curl_normb_y(i, k)
                D_par_par   = D_par_par   * bps
                D_perp_par  = D_perp_par  * bps
                D_perp_perp = D_perp_perp * bps
                E_par       = E_par       * bps
                E_perp      = E_perp      * bps

                ! Calculate the collisional fluxes using the FPL coefficients

                ! NOTE: We need to use f_buffer here since for multiple
                !       processes in vp and mu the ghost value of the diagonal
                !       neighbour is not exchanged in f_in.
                f00 = f_buffer(i, k, l    , m    , n)
                f0p = f_buffer(i, k, l    , m + 1, n)
                fp0 = f_buffer(i, k, l + 1, m    , n)
                fpp = f_buffer(i, k, l + 1, m + 1, n)

                delta_par_p  = delta_par(l)
                delta_perp_p = delta_perp(m)
                delta_par_m  = 1.0_GP - delta_par_p
                delta_perp_m = 1.0_GP - delta_perp_p

                finterp = delta_par_m * delta_perp_m * f00 &
                        + delta_par_m * delta_perp_p * f0p &
                        + delta_par_p * delta_perp_m * fp0 &
                        + delta_par_p * delta_perp_p * fpp

                fdiff_par  = ( delta_perp_p * fpp &
                             - delta_perp_p * f0p &
                             + delta_perp_m * fp0 &
                             - delta_perp_m * f00 ) * inv_dvpar

                fdiff_perp = ( delta_par_p * fpp &
                             - delta_par_p * fp0 &
                             + delta_par_m * f0p &
                             - delta_par_m * f00 ) * inv_dvperp

                ! Perpendicular flux
                flux_perp(l, m) = flux_perp(l, m) + E_perp * finterp &
                                + D_perp_perp * fdiff_perp &
                                + D_perp_par  * fdiff_par

                ! Parallel flux
                flux_par(l, m) = flux_par(l, m) + E_par * finterp &
                               + D_perp_par * fdiff_perp &
                               + D_par_par * fdiff_par
            enddo
            enddo

            ! Apply collisions
            do m = lb_stripped(4), ub_stripped(4)
            do l = lb_stripped(3), ub_stripped(3)

                ! Upper and lower area element ratio
                area_ratio_p = this%sqrt_mu_stag(m    ) / sqrt_mu(m)
                area_ratio_m = this%sqrt_mu_stag(m - 1) / sqrt_mu(m)

                ! NOTE: In this finite volume scheme the perpendicular outgoing
                !       flux of a cell is not the same as the incoming flux of
                !       the next cell due to the volume element which increases
                !       in the perpendicular direction. Multiplying the second
                !       contribution by (m-1/2)/(m-3/2) accounts for that
                !       difference.
                df_perp1 = area_ratio_p &
                         * (flux_perp(l, m    ) + flux_perp(l - 1, m    ))
                df_perp2 =-area_ratio_m &
                         * (flux_perp(l, m - 1) + flux_perp(l - 1, m - 1))

                ! NOTE: The parallel fluxes are collected by their corresponding
                !       area prefactors. To get the real fluxes on the side of a
                !       cell, on must consider the contributions with same l.
                df_par1  = area_ratio_p &
                         * (flux_par(l, m    ) - flux_par(l - 1, m    ))
                df_par2  = area_ratio_m &
                         * (flux_par(l, m - 1) - flux_par(l - 1, m - 1))

                ! Apply B parallel star (=Jacobian) from phase space divergence
                ! (outer part)
                bps  = absB(i, k) &
                     + this%prefac_bps(n) * vp(l) * curl_normb_y(i, k)

                f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                     - 0.5_GP / bps &
                                        * ( inv_dvperp * (df_perp1 + df_perp2) &
                                          + inv_dvpar  * (df_par1  + df_par2 ) )

                ! Apply compute mask
                f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                     * is_compute(i, k)
            enddo
            enddo
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("compute", ierr)

        deallocate(f_send, f_buffer)
        deallocate(flux_perp, flux_par)

        ! Error handling

        ! NOTE: For the interpolation, we do not consider the case of no root
        !       between 0 and 1 as fatal yet. The default 1/2 will be used
        !       in that case. Thus we only throw a warning.
        if(err_interp) then
            call handle_error("Interpolation scheme is ill-posed, at least &
                              &one pair of roots was out of range &
                              &(in FPL collisions)!", &
                              GENEX_WRN_COLL, __LINE__, __FILE__)
        endif
        ! Invalid argument in elliptic integrals is fatal
        if(err_ceik) then
            call handle_error("Fukushima ceik encountered values for argument &
                              &m which are out of range [0, 1]!", &
                              GENEX_WRN_COLL, __LINE__, __FILE__)
        endif
        if(err_ceie) then
            call handle_error("Fukushima ceie encountered values for argument &
                              &m which are out of range [0, 1]!", &
                              GENEX_WRN_COLL, __LINE__, __FILE__)
        endif
        if(err_ceik .or. err_ceie) then
            call handle_error("Failed to evaluate elliptic integrals in FPL &
                              &velocity tensor!", &
                              GENEX_ERR_COLL, __LINE__, __FILE__)
        endif

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine initialize_staggered_bounds(this, lb_stripped, &
                                                  ub_stripped)
        class(op_coll_fpl_cpu_t), intent(inout) :: this
        integer, dimension(5), intent(in) :: lb_stripped, ub_stripped

            ! The bounds of the staggered grid depend on the bounds of the
            ! current MPI process. On the domain boundary we need one less
            ! point on the staggered grid. Intra-domain boundaries need to
            ! overlap one staggered grid point.

            if(lb_stripped(3) == 1) then
                this%lb_stagger(3) = lb_stripped(3)
            else
                this%lb_stagger(3) = lb_stripped(3) - 1
            endif
            if(ub_stripped(3) == this%n_vp) then
                this%ub_stagger(3) = ub_stripped(3) - 1
            else
                this%ub_stagger(3) = ub_stripped(3)
            endif
            if(lb_stripped(4) == 1) then
                this%lb_stagger(4) = lb_stripped(4)
            else
                this%lb_stagger(4) = lb_stripped(4) - 1
            endif
            if(ub_stripped(4) == this%n_mu) then
                this%ub_stagger(4) = ub_stripped(4) - 1
            else
                this%ub_stagger(4) = ub_stripped(4)
            endif
    end subroutine

    pure module subroutine calc_interp_weights(this, inv_temp, absB, &
                                               delta_par, delta_perp, ierr)
        class(op_coll_fpl_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: inv_temp
        real(kind=GP), intent(in) :: absB
        real(kind=GP), dimension(1:this%n_vp), intent(out) :: delta_par
        real(kind=GP), dimension(1:this%n_mu), intent(out) :: delta_perp
        integer, intent(out) :: ierr

        integer :: l, m
        ! Loop variables
        real(kind=GP) :: alpha
        ! Argument alpha (negative capital lambda in Hager 2016)
        real(kind=GP) :: root1, root2
        ! First and second root of delta equation

        ! Calculates the interpolation weights for the distribution function
        ! such that the thermal equilibrium is conserved.
        ! See Yoon 2014 eq. 19 and Hager 2016 eq. 25

        ierr = GENEX_SUCCESS

        delta_par  = 0.5_GP
        delta_perp = 0.5_GP

        do l = 1, this%n_vp
            ! Skip the point which yields ill-defined roots
            if(abs(this%vp_stag(l)) < zero_tol) cycle

            alpha = this%delta_vp * this%vp_stag(l) * inv_temp
            root2 = 1.0_GP / (1.0_GP - exp(alpha))
            root1 = root2 + 1.0_GP / alpha

            ! Choose the root which is between 0 and 1
            ! NOTE: 1 - root instead of the root itself is the correct
            !       value that results in entropy increase.
            if(root1 >= 0.0_GP .and. root1 <= 1.0_GP) then
                delta_par(l) = 1.0_GP - root1
            elseif(root2 >= 0.0_GP .and. root2 <= 1.0_GP) then
                delta_par(l) = 1.0_GP - root2
            else
                ierr = GENEX_ERR_COLL
            endif
        enddo

        do m = 1, this%n_mu
            ! Skip the point which yields ill-defined roots
            if(abs(this%sqrt_mu_stag(m)) < zero_tol) cycle

            alpha = this%delta_sqrt_mu * this%sqrt_mu_stag(m) &
                  * absB * inv_temp
            root2 = 1.0_GP / (1.0_GP - exp(alpha))
            root1 = root2 + 1.0_GP / alpha

            ! Choose the root which is between 0 and 1
            ! NOTE: 1 - root instead of the root itself is the correct
            !       value that results in entropy increase.
            if(root1 >= 0.0_GP .and. root1 <= 1.0_GP) then
                delta_perp(m) = 1.0_GP - root1
            elseif(root2 >= 0.0_GP .and. root2 <= 1.0_GP) then
                delta_perp(m) = 1.0_GP - root2
            else
                ierr = GENEX_ERR_COLL
            endif
        enddo
    end subroutine

    pure function fukushima_ceik_local(m_arg) result(res)
        !! Local wrapper for calculating the elliptic integral K(m). Used so
        !! that the compiler can optimize the function.
        !! See elliptic_integrals module for full description and tests.
        real(kind=GP), intent(in) :: m_arg
        !! Argument to evaluate K
        real(kind=GP) :: res
        !! Result

        real(kind=DP) :: mc, ceik, t, t2

        ! NOTE: The complementary argument mc is used in the algorithm
        mc = 1.0_DP - m_arg

        ! NOTE: We avoid error handling and I/O for optimization. Instead the
        !       function will return zero in case of an error. This is a value
        !       that should not be present in the interval [0, 1].

#include "../../math/fukushima_ceik.txt"

        res = real(ceik, kind=GP)
    end function

    pure function fukushima_ceie_local(m_arg) result(res)
        !! Local wrapper for calculating the elliptic integral E(m)
        real(kind=GP), intent(in) :: m_arg
        !! Argument to evaluate E
        real(kind=GP) :: res
        !! Result

        real(kind=DP) :: mc, ceie, t, t2

        ! NOTE: The complementary argument mc is used in the algorithm
        mc = 1.0_DP - m_arg

        ! NOTE: We avoid error handling and I/O for optimization. Instead the
        !       function will return zero in case of an error. This is a value
        !       that should not be present in the interval [0, 1].

#include "../../math/fukushima_ceie.txt"

        res = real(ceie, kind=GP)
    end function

end submodule
