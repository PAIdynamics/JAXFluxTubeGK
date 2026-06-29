submodule (mesh_5d_m) mesh_5d_magfield_s
    !! This module contains the implementation of the magnetic field and its
    !! derivatives, as well as poloidal flux psi, flux surface label rho, and
    !! poloidal angle theta.
    use genex_fortran_env_m, only: DP
    use math_m, only : PI
    use params_normalization_m, only : get_L_ref, get_rho_ref, get_B_ref
    use params_species_m, only : get_mass, get_charge, get_temp_scaling
    use logger_m, only: logger_get_info_channel
    use params_initial_condition_m, only: get_use_canonical_maxw
    use params_diagnostics_m, only: get_polar_n_theta, get_polar_n_rho, &
                                                                get_diagnose_tpc
    use profiler_m, only: profiler_start, profiler_stop
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use fieldline_tracer_m, only : trace
    use csrmat_m, only : csrmat_t, csr_times_vec
    use polar_map_factory_m, only: &
                                 create_polar_map_matrix, create_cart_map_matrix

    implicit none

contains

    module subroutine initialize_magfield(this)
        class(mesh_5d_t), intent(inout) :: this
        integer :: ierr

        call profiler_start("initialize_B", ierr, set_path=.true.)
        call initialize_B(this)
        call profiler_stop("initialize_B", ierr, set_path =.true.)
        call profiler_start("initialize_dB", ierr, set_path=.true.)
        call initialize_dB(this)
        call profiler_stop("initialize_dB", ierr, set_path=.true.)

        call profiler_start("initialize_rho", ierr, set_path=.true.)
        call initialize_rho_theta(this)
        call initialize_canonical_rho(this)
        call profiler_stop("initialize_rho", ierr, set_path=.true.)

        if (get_diagnose_tpc()) then
            call initialize_B_pol(this)
            call initialize_loss_cone(this)
        endif

    end subroutine

    real(kind=GP) module function canonical_rho(this, i, k, l, m, n)
        class(mesh_5d_t), intent(inout) :: this
        integer, intent(in) :: i, k, l, m, n

        ! Canonical rho is only available for toroidal equilibria with gradient
        ! in B. For others return ordinary rho.
        if(this%equi_type() == SALPHA .or. this%equi_type() == NUMERICAL) then
            ! Calculate rho normalized w.r.t. psi_sep and psi_ax (not cano!)
            canonical_rho = (this%canonical_psi(i, k, l, m, n) - this%psi_lo) &
                          / (this%psi_up - this%psi_lo)
            if(canonical_rho .gt. 0.0_GP) then
                ! Calculate square root only if larger than zero (should be)
                ! but do this to prevent NaNs
                canonical_rho = sqrt(canonical_rho)
            else
                ! Otherwise set to zero (this is similar to rho in parallax)
                canonical_rho = 0.0_GP
            end if
        else
            canonical_rho = this%rho_buffer(i, k)
        end if
    end function

    real(kind=GP) module function canonical_psi(this, i, k, l, m, n)
        class(mesh_5d_t), intent(inout) :: this
        integer, intent(in) :: i, k, l, m, n

        real(kind=GP) :: x, y, phi, psi, vp_bar, energy_diff, psi_corr
        real(kind=GP), dimension(:), contiguous, pointer :: vp, mu
        integer, dimension(:), contiguous, pointer :: bsg_flags
        integer :: nb_point

        bsg_flags => this%get_bsg_flags_pointer()

        nb_point = unpack_bflag(bsg_flags(i))
        vp => this%vp_grid(nb_point)%get_pointer()
        mu => this%mu_grid%get_pointer()

        ! Energy difference correction
        energy_diff = vp(l)**2 + mu(m) &
                               * (this%absB_buffer(i, k) - this%absB_max)
        if(energy_diff > 0.0_GP) then
            vp_bar = sign(1.0_GP, vp(l)) * sqrt(energy_diff)
        else
            vp_bar = 0.0_GP
        endif

        ! Calculate correction to psi that must be added to get canonical psi
        psi_corr = get_rho_ref() / get_L_ref() / get_charge(n) &
                 * sqrt(2.0_GP * get_mass(n) * get_temp_scaling(n)) &
                 * (this%normb_tor_buffer(i, k) * this%R_buffer(i, k) * vp(l) &
                   - vp_bar)

        call this%RZphi(i, k, x, y, phi)

        ! Psi only defined for salpha or divertor equilibria.
        ! Set to zero in the other cases.
        select type(equilibrium_instance)
        type is(salpha_t)
            psi = equilibrium_instance%psi(x, y) / this%psi_norm_fac
            canonical_psi = psi + psi_corr
        type is(numerical_t)
            psi = equilibrium_instance%psi(x, y, phi) / this%psi_norm_fac
            canonical_psi = psi + psi_corr
        class default
            ! This should not happen, give warning.
            canonical_psi = 0.0_GP
            write(logger_get_info_channel(), *) &
                "Info: canonical_psi called for inappropriate equilibrium type,&
                    & zero returned!"
        end select
    end function

    subroutine initialize_B(this)
        !! Initializes the magnetic field and psi buffers
        class(mesh_5d_t), intent(inout) :: this
        !! Instance of the mesh

        integer :: i, k, n_points_RZ, n_points_phi
        real(kind=GP) :: x, z, phi, norm_psi
        real(kind=GP), allocatable, dimension(:,:) :: psi_buffer, absB_buffer

        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()

        norm_psi = 1.0_GP / (get_L_ref()**2 * get_B_ref() * 2 * PI)

        allocate(absB_buffer(n_points_RZ, n_points_phi))
        allocate(psi_buffer(n_points_RZ, n_points_phi))

        do k = 1, n_points_phi
        do i = 1, n_points_RZ
            if(this%not_filler(i, k) == 1.0_GP) then
                call this%RZphi(i, k, x, z, phi)
                absB_buffer(i, k) = equilibrium_instance%absB(x, z, phi)
                select type(equilibrium_instance)
                type is(numerical_t)
                    psi_buffer(i, k) = norm_psi &
                                     * equilibrium_instance%psi(x, z, phi)
                type is(cerfons_t)
                    ! TODO: Check psi for cerfons
                    psi_buffer(i, k) = norm_psi &
                                     * equilibrium_instance%psi(x, z, phi)
                type is(salpha_t)
                    psi_buffer(i, k) = norm_psi &
                                     * equilibrium_instance%psi(x, z)
                type is(circular_t)
                    ! TODO: Implement psi for circular
                    psi_buffer(i, k) = 0.0_GP
                class default
                    psi_buffer(i, k) = 0.0_GP
                end select
            else
                ! Filler point
                ! NOTE: The filler value for absB must be positive to avoid
                !       errors when taking the square-root
                absB_buffer(i, k) = abs(FILLER_VALUE_REAL)
                psi_buffer(i, k)  = FILLER_VALUE_REAL
            endif
        enddo
        enddo

        ! First touch initialize the magnetic field and poloidal flux function
        ! on the mesh_3d
        allocate(this%absB_buffer, mold=absB_buffer)
        allocate(this%psi_buffer,  mold=absB_buffer)
        !$omp parallel default(none) &
        !$omp private(i, k) firstprivate(n_points_RZ, n_points_phi) &
        !$omp shared(this, absB_buffer, psi_buffer)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            this%absB_buffer(i, k) = absB_buffer(i, k)
            this%psi_buffer(i, k) = psi_buffer(i, k)
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel
        deallocate(psi_buffer, absB_buffer)
    end subroutine

    subroutine initialize_B_pol(this)
        !! Initializes the magnetic field on a polar mesh
        class(mesh_5d_t), intent(inout) :: this
        !! Instance of the mesh

        integer :: p_i, p_j, k, n_points_phi, polar_n_theta, polar_n_rho
        real(kind=GP) :: phi
        real(kind=GP), allocatable, dimension(:,:,:) :: polar_absB_buffer
        type(csrmat_t), allocatable :: polar_map

        n_points_phi = this%size_phi()

        polar_n_theta = get_polar_n_theta()
        polar_n_rho = get_polar_n_rho()

        allocate(this%mesh_3d_pol(n_points_phi))
        allocate(polar_absB_buffer(polar_n_theta, polar_n_rho, n_points_phi))

        do k = 1, n_points_phi
            phi = this%mesh_3d(modulo(k - 1, this%size_phi()) + 1)%get_phi()
            call this%mesh_3d_pol(k)%initialize(equilibrium_instance, phi, &
                                                polar_n_rho, polar_n_theta)
            call create_polar_map_matrix(polar_map, &
                                         equilibrium_instance, &
                                         this%mesh_3d(k), this%mesh_3d_pol(k), &
                                         polar_intorder)
            call csr_times_vec(polar_map, this%absB_buffer(:, k), &
                               polar_absB_buffer(:, :, k))
        enddo

        allocate(this%polar_absB_buffer, mold=polar_absB_buffer)
        !$omp parallel default(none) &
        !$omp private(p_i, p_j, k) &
        !$omp firstprivate(polar_n_rho, polar_n_theta, n_points_phi) &
        !$omp shared(this, polar_absB_buffer)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do p_j = 1, polar_n_rho
        do p_i = 1, polar_n_theta
            this%polar_absB_buffer(p_i, p_j, k) = polar_absB_buffer(p_i, p_j, k)
        enddo
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel
        deallocate(polar_absB_buffer)
    end subroutine

    subroutine initialize_dB(this)
        !! Initializes the buffers containing the derivatives of the magnetic
        !! field and the components of the metric tensor
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the mesh

        real(kind=GP) :: h, delta_phi
        real(kind=GP) :: xc, zc, phi, &
                         g, dgyzdy, dgyxdy, dgyzdx, dgyxdz, &
                         b_plus, b_minus, &
                         xc_plus, xc_minus, &
                         yc_plus, yc_minus, &
                         zc_plus, zc_minus, &
                         R_plus, R_minus, &
                         phi_plus, phi_minus, &
                         ds_plus, ds_minus
        real(kind=GP), allocatable, dimension(:,:) :: normb_tor_buffer, &
                                                      normb_R_buffer, &
                                                      normb_Z_buffer, &
                                                      dgyxdy_over_g, &
                                                      dgyzdy_over_g, &
                                                      dgyxdz_over_g, &
                                                      dgyzdx_over_g, &
                                                      inv_g, &
                                                      dabsBdy, &
                                                      dabsBdz, &
                                                      dabsBdx, &
                                                      curl_normb_y
        integer :: n_points_RZ, n_points_phi, i, k

        if(GP == DP) then
            h = 1e-5_GP
            delta_phi = 2 * PI * 1e-6_GP
        else
            h = 1e-4_GP
            delta_phi = 2 * PI * 1e-3_GP
        endif

        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()

        allocate(normb_tor_buffer(n_points_RZ, n_points_phi))

        allocate(normb_R_buffer, mold=normb_tor_buffer)
        allocate(normb_Z_buffer, mold=normb_tor_buffer)
        allocate(dgyxdy_over_g,  mold=normb_tor_buffer)
        allocate(dgyzdy_over_g,  mold=normb_tor_buffer)
        allocate(dgyxdz_over_g,  mold=normb_tor_buffer)
        allocate(dgyzdx_over_g,  mold=normb_tor_buffer)
        allocate(inv_g,          mold=normb_tor_buffer)
        allocate(dabsBdy,        mold=normb_tor_buffer)
        allocate(dabsBdz,        mold=normb_tor_buffer)
        allocate(dabsBdx,        mold=normb_tor_buffer)
        allocate(curl_normb_y,   mold=normb_tor_buffer)

        allocate(this%normb_tor_buffer, mold=normb_tor_buffer)
        allocate(this%normb_R_buffer,   mold=normb_tor_buffer)
        allocate(this%normb_Z_buffer,   mold=normb_tor_buffer)
        allocate(this%dgyxdy_over_g,    mold=normb_tor_buffer)
        allocate(this%dgyzdy_over_g,    mold=normb_tor_buffer)
        allocate(this%dgyxdz_over_g,    mold=normb_tor_buffer)
        allocate(this%dgyzdx_over_g,    mold=normb_tor_buffer)
        allocate(this%inv_g,            mold=normb_tor_buffer)
        allocate(this%dabsBdy,          mold=normb_tor_buffer)
        allocate(this%dabsBdz,          mold=normb_tor_buffer)
        allocate(this%dabsBdx,          mold=normb_tor_buffer)
        allocate(this%curl_normb_y,     mold=normb_tor_buffer)

        !$omp parallel default(none) &
        !$omp private(i, k, xc, zc, phi, &
        !$omp         g, dgyzdy, dgyxdy, dgyzdx, dgyxdz, &
        !$omp         b_plus, b_minus, xc_plus, xc_minus, yc_plus, yc_minus, &
        !$omp         zc_plus, zc_minus, R_plus, R_minus, phi_plus, phi_minus, &
        !$omp         ds_plus, ds_minus) &
        !$omp firstprivate(n_points_RZ, n_points_phi, h, delta_phi) &
        !$omp shared(this, normb_tor_buffer, normb_R_buffer, normb_Z_buffer, &
        !$omp        inv_g, dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, &
        !$omp        dgyzdx_over_g, dabsBdy, dabsBdz, dabsBdx, curl_normb_y, &
        !$omp        equilibrium_instance)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
        if(this%not_filler(i, k) == 1.0_GP) then

            call this%RZphi(i, k, xc, zc, phi)

            ! Since the cartesian coodinate system is local to each plane, on
            ! the plane the yc coordinate is always zero. We have to pass in
            ! the global phi for the 3D magnetic field.
            normb_R_buffer(i, k)   = norm_bxc(xc, 0.0_GP, zc, phi)
            normb_Z_buffer(i, k)   = norm_bzc(xc, 0.0_GP, zc, phi)
            normb_tor_buffer(i, k) = norm_byc(xc, 0.0_GP, zc, phi)

            g = abs(normb_tor_buffer(i, k))
            inv_g(i, k) = 1.0_GP / g

            ! Calculation of curl of b
            b_plus  = norm_bzc(xc + h, 0.0_GP, zc, phi)
            b_minus = norm_bzc(xc - h, 0.0_GP, zc, phi)
            dgyzdx  = (b_plus - b_minus) / (2 * h)

            b_plus  = norm_bxc(xc, 0.0_GP, zc + h, phi)
            b_minus = norm_bxc(xc, 0.0_GP, zc - h, phi)
            dgyxdz  = (b_plus - b_minus) / (2 * h)

            b_plus  = equilibrium_instance%absB(xc + h, zc, phi)
            b_minus = equilibrium_instance%absB(xc - h, zc, phi)
            dabsBdx(i, k) = (b_plus - b_minus) / (2 * h)

            b_plus  = equilibrium_instance%absB(xc, zc + h, phi)
            b_minus = equilibrium_instance%absB(xc, zc - h, phi)
            dabsBdz(i, k) = (b_plus - b_minus) / (2 * h)

            call trace(xc, zc, phi, delta_phi, equilibrium_instance, &
                       x_end=R_plus, y_end=zc_plus, phi_end=phi_plus, &
                       arclen=ds_plus)
            call trace(xc, zc, phi, -delta_phi, equilibrium_instance, &
                       x_end=R_minus, y_end=zc_minus, phi_end=phi_minus, &
                       arclen=ds_minus)
            ! The trace returns the R and Z coordinate in a cylindrical
            ! coordinate system, although it is called x and y. We convert to
            ! cartesian coordinates below.
            if(this%equi_type() == SLAB .or. this%equi_type() == CIRCULAR) then
                xc_plus  = R_plus
                yc_plus  = delta_phi
                xc_minus = R_minus
                yc_minus = -delta_phi
            else
                xc_plus  = R_plus * cos(delta_phi)
                yc_plus  = R_plus * sin(delta_phi)
                xc_minus = R_minus * cos(-delta_phi)
                yc_minus = R_minus * sin(-delta_phi)
            endif

            b_plus  = norm_bzc(xc_plus, yc_plus, zc_plus, phi)
            b_minus = norm_bzc(xc_minus, yc_minus, zc_minus, phi)
            dgyzdy  = (b_plus - b_minus) / (ds_plus + ds_minus)

            b_plus  = norm_bxc(xc_plus, yc_plus, zc_plus, phi)
            b_minus = norm_bxc(xc_minus, yc_minus, zc_minus, phi)
            dgyxdy  = (b_plus - b_minus) / (ds_plus + ds_minus)

            b_plus  = equilibrium_instance%absB(R_plus, zc_plus, phi_plus)
            b_minus = equilibrium_instance%absB(R_minus, zc_minus, phi_minus)
            dabsBdy(i, k) = (b_plus - b_minus) / (ds_plus + ds_minus)

            curl_normb_y(i, k) = 1.0_GP / g &
                               * ( (dgyzdy         ) * normb_R_buffer(i, k) &
                                 + (       - dgyxdy) * normb_Z_buffer(i, k) &
                                 + (dgyxdz - dgyzdx) * 1.0_GP)

            dgyxdz_over_g(i, k) = dgyxdz / g
            dgyzdx_over_g(i, k) = dgyzdx / g
            dgyxdy_over_g(i, k) = dgyxdy / g
            dgyzdy_over_g(i, k) = dgyzdy / g
        else
            ! Filler points
            normb_tor_buffer(i, k) = FILLER_VALUE_REAL
            normb_R_buffer(i, k)   = FILLER_VALUE_REAL
            normb_Z_buffer(i, k)   = FILLER_VALUE_REAL
            dgyxdz_over_g(i, k)    = FILLER_VALUE_REAL
            dgyzdx_over_g(i, k)    = FILLER_VALUE_REAL
            dgyxdy_over_g(i, k)    = FILLER_VALUE_REAL
            dgyzdy_over_g(i, k)    = FILLER_VALUE_REAL
            dabsBdy(i, k)          = FILLER_VALUE_REAL
            dabsBdz(i, k)          = FILLER_VALUE_REAL
            dabsBdx(i, k)          = FILLER_VALUE_REAL
            curl_normb_y(i, k)     = FILLER_VALUE_REAL
            inv_g(i, k)            = FILLER_VALUE_REAL
        endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        ! First touch initialize
        !$omp parallel default(none) &
        !$omp private(i, k) firstprivate(n_points_RZ, n_points_phi) &
        !$omp shared(this, normb_tor_buffer, normb_R_buffer, normb_Z_buffer, &
        !$omp        inv_g, dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, &
        !$omp        dgyzdx_over_g, dabsBdy, dabsBdz, dabsBdx, curl_normb_y)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            this%normb_tor_buffer(i, k) = normb_tor_buffer(i, k)
            this%normb_R_buffer(i, k)   = normb_R_buffer(i, k)
            this%normb_Z_buffer(i, k)   = normb_Z_buffer(i, k)
            this%dgyxdy_over_g(i, k)    = dgyxdy_over_g(i, k)
            this%dgyzdy_over_g(i, k)    = dgyzdy_over_g(i, k)
            this%dgyxdz_over_g(i, k)    = dgyxdz_over_g(i, k)
            this%dgyzdx_over_g(i, k)    = dgyzdx_over_g(i, k)
            this%dabsBdy(i, k)          = dabsBdy(i, k)
            this%dabsBdz(i, k)          = dabsBdz(i, k)
            this%dabsBdx(i, k)          = dabsBdx(i, k)
            this%curl_normb_y(i, k)     = curl_normb_y(i, k)
            this%inv_g(i, k)            = inv_g(i, k)
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        deallocate(normb_tor_buffer, normb_R_buffer, normb_Z_buffer, inv_g, &
                   dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, &
                   dgyzdx_over_g, dabsBdy, dabsBdz, dabsBdx, curl_normb_y)
    contains

        real(kind=GP) function norm_bxc(xc, yc, zc, phi)
            !! Calculates the x component of the normalized magnetic field
            !! in a cartesian coordinate system centered on the poloidal plane
            !! at angle phi
            real(kind=GP), intent(in) :: xc, yc, zc, phi

            real(kind=GP) :: R, phi_prime
            ! Cylindrical coordinates for PARALLAX

            if(this%equi_type() == SLAB .or. this%equi_type() == CIRCULAR) then
                ! For SLAB and CIRCULAR the magnetic field is provided in
                ! cartesian coordinates by PARALLAX. No transformations are
                ! necessary for x and z, while y corresponds to dphi
                phi_prime = phi + yc
                norm_bxc = equilibrium_instance%Bx(xc, zc, phi_prime) &
                         / equilibrium_instance%absB(xc, zc, phi_prime)
            else
                ! For other equilibra the magnetic field is provided in
                ! cylindrical coordinates by PARALLAX. We transform to
                ! cartesian coordinates
                R = Sqrt(xc**2 + yc**2)
                phi_prime = phi + atan2(yc, xc)
                norm_bxc = ( xc * equilibrium_instance%Bx(R, zc, phi_prime) &
                           - yc * equilibrium_instance%Btor(R, zc, phi_prime)) &
                         / ( R * equilibrium_instance%absB(R, zc, phi_prime))
            endif
        end function

        real(kind=GP) function norm_byc(xc, yc, zc, phi)
            !! Calculates the y component of the normalized magnetic field
            !! in a cartesian coordinate system centered on the poloidal plane
            !! at angle phi
            real(kind=GP), intent(in) :: xc, yc, zc, phi

            real(kind=GP) :: R, phi_prime
            ! Cylindrical coordinates for PARALLAX

            if(this%equi_type() == SLAB .or. this%equi_type() == CIRCULAR) then
                ! For SLAB and CIRCULAR the magnetic field is provided in
                ! cartesian coordinates by PARALLAX. No transformations are
                ! necessary for x and z, while y corresponds to dphi
                phi_prime = phi + yc
                norm_byc = equilibrium_instance%Btor(xc, zc, phi_prime) &
                         / equilibrium_instance%absB(xc, zc, phi_prime)
            else
                ! For other equilibra the magnetic field is provided in
                ! cylindrical coordinates by PARALLAX. We transform to
                ! cartesian coordinates
                R = Sqrt(xc**2 + yc**2)
                phi_prime = phi + atan2(yc, xc)
                norm_byc = ( yc * equilibrium_instance%Bx(R, zc, phi_prime) &
                           + xc * equilibrium_instance%Btor(R, zc, phi_prime)) &
                         / ( R * equilibrium_instance%absB(R, zc, phi_prime))
            endif
        end function

        real(kind=GP) function norm_bzc(xc, yc, zc, phi)
            !! Calculates the z component of the normalized magnetic field
            !! in a cartesian coordinate system centered on the poloidal plane
            !! at angle phi
            real(kind=GP), intent(in) :: xc, yc, zc, phi

            real(kind=GP) :: R, phi_prime
            ! Cylindrical coordinates for PARALLAX

            if(this%equi_type() == SLAB .or. this%equi_type() == CIRCULAR) then
                ! For SLAB and CIRCULAR the magnetic field is provided in
                ! cartesian coordinates by PARALLAX. No transformations are
                ! necessary for x and z, while y corresponds to dphi
                phi_prime = phi + yc
                norm_bzc = equilibrium_instance%By(xc, zc, phi_prime) &
                         / equilibrium_instance%absB(xc, zc, phi_prime)
            else
                ! For other equilibra the magnetic field is provided in
                ! cylindrical coordinates by PARALLAX. We transform to
                ! cartesian coordinates
                R = Sqrt(xc**2 + yc**2)
                phi_prime = phi + atan2(yc, xc)
                norm_bzc = equilibrium_instance%By(R, zc, phi_prime) &
                         / equilibrium_instance%absB(R, zc, phi_prime)
            endif

        end function
    end subroutine

    subroutine initialize_rho_theta(this)
        !! Initializes the flux surface label rho and the poloidal angle on the
        !! 3D grid.
        !! For slab, theta is the Z coordinate. In all other cases, theta is the
        !! geometric poloidal angle centered around the magnetic axis.
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the mesh

        integer :: i, k, n_points_RZ, n_points_phi
        real(kind=GP), dimension(:), pointer :: phi
        real(kind=GP) :: x, y, axis_x, axis_y

        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()
        phi => this%get_phi_pointer()

        allocate(this%rho_buffer(n_points_RZ, n_points_phi))
        allocate(this%theta_buffer(n_points_RZ, n_points_phi))

        do k = 1, n_points_phi
            call equilibrium_instance%mag_axis_loc(phi(k), axis_x, axis_y)
        do i = 1, n_points_RZ
        if(this%not_filler(i, k) == 1.0_GP) then
            x = this%R_buffer(i, k)
            y = this%Z_buffer(i, k)

            this%rho_buffer(i, k) = equilibrium_instance%rho(x, y, phi(k))

            select type(equilibrium_instance)
            type is(slab_t)
                this%theta_buffer(i, k) = y
            class default
                this%theta_buffer(i, k) = atan2(y - axis_y, x - axis_x)
            end select
        else
            this%rho_buffer(i, k) = FILLER_VALUE_REAL
            this%theta_buffer(i, k) = FILLER_VALUE_REAL
        end if
        end do
        end do

    end subroutine

    subroutine initialize_canonical_rho(this)
        !! Initializes the canonical flux surface label
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the mesh

        integer :: i, k, l, m, n
        real(kind=GP) :: x, y, phi, absB, psi_c

        ! Search for maximum value of absB
        this%absB_max = 0.0_GP
        !$omp parallel default(none) &
        !$omp shared(this, equilibrium_instance) &
        !$omp private(i, k, x, y, phi, absB)
        do k = 1, this%size_phi()
        !$omp do schedule(static)
        do i = 1, this%size_RZ()
        if(this%not_filler(i, k) == 1.0_GP) then
            call this%RZphi(i, k, x, y, phi)
            absB = equilibrium_instance%absB(x, y, phi)
            if(absB > this%absB_max) this%absB_max = absB
        end if
        end do
        !$omp end do
        end do
        !$omp end parallel

        ! Canonical rho can only be computed for salpha or divertor equilibria
        ! because only these type/class contains psi.
        ! Normalization can only be done for numerical equilibria in the latter
        ! case because of the variables R0 and axis_Btor.

        ! Calculate psi at axis and separatrix
        select type(equilibrium_instance)
        type is(salpha_t)
            ! Normalize w.r.t. to reference length and magnetic field
            this%psi_norm_fac = 2.0_GP * PI * get_L_ref()**2 * get_B_ref()

            ! Set the upper and lower values of the rho normalization
            ! to the min/max value of psi_c doing a simple search
            this%psi_lo =  1e100_GP
            this%psi_up = -1e100_GP
            !$omp parallel default(none) &
            !$omp shared(this) &
            !$omp private(n, m, l, k, i, psi_c)
            do n = 1, this%size_sp()
            do m = 1, this%size_mu()
            do l = 1, this%size_vp()
            do k = 1, this%size_phi()
            !$omp do schedule(static)
            do i = 1, this%size_RZ()
                psi_c = this%canonical_psi(i, k, l, m, n)
                if(psi_c < this%psi_lo) this%psi_lo = psi_c
                if(psi_c > this%psi_up) this%psi_up = psi_c
            end do
            !$omp end do
            end do
            end do
            end do
            end do
            !$omp end parallel
        type is(numerical_t)
            ! For numerical equilibria we need to normalize w.r.t. the major
            ! radius R0 and toroidal field at R0 that is contained in the
            ! equilibrium
            this%psi_norm_fac = 2.0_GP * PI * &
                equilibrium_instance%R0**2 * equilibrium_instance%axis_Btor

            ! Set the upper and lower values of the rho normalization
            ! to the value of psi at the separatrix and magnetic axis
            this%psi_lo = equilibrium_instance%o_point_psi / this%psi_norm_fac
            this%psi_up = equilibrium_instance%x_point_psi / this%psi_norm_fac
        class default
            ! Give a single info here that canonical rho was called for not
            ! supported equilibria. The function will return the ordinary rho
            ! in that case.
            if(get_use_canonical_maxw()) then
                write(logger_get_info_channel(), *) &
                    "Info: canonical Maxwellian used for not supported&
                        & equilibrium type. Used ordinary rho instead!"
            end if
        end select
    end subroutine

    subroutine initialize_loss_cone(this)
        !! Initializes the TPC loss cone
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the mesh

        integer :: i, p_i, p_j, k, polar_n_theta, polar_n_rho, n_points_phi, &
                   n_points_RZ
        real(kind=GP) :: max_y_B
        real(kind=GP), allocatable, dimension(:,:) :: loss_cone
        real(kind=GP), allocatable, dimension(:,:,:) :: loss_cone_pol
        type(csrmat_t), allocatable :: matrix_3d_cart

        polar_n_theta = get_polar_n_theta()
        polar_n_rho   = get_polar_n_rho()
        n_points_phi  = this%size_phi()
        n_points_RZ   = this%size_RZ()

        allocate(loss_cone(n_points_RZ, n_points_phi))
        allocate(loss_cone_pol(polar_n_theta, polar_n_rho, n_points_phi))
        !$omp parallel default(none) &
        !$omp private(p_i, p_j, k, max_y_B) &
        !$omp firstprivate(polar_n_theta, polar_n_rho, n_points_phi) &
        !$omp shared(this, loss_cone_pol)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do p_j = 1, polar_n_rho
            max_y_B = maxval(this%polar_absB_buffer(:, p_j, k))
            do p_i = 1, polar_n_theta
                loss_cone_pol(p_i, p_j, k) = max_y_B &
                                           - this%polar_absB_buffer(p_i, p_j, k)
            enddo
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        do k = 1, n_points_phi
            call create_cart_map_matrix(matrix_3d_cart, equilibrium_instance, &
                                        this%mesh_3d(k), this%mesh_3d_pol(k), &
                                        polar_intorder)
            call csr_times_vec(matrix_3d_cart, loss_cone_pol(:, :, k), &
                               loss_cone(:, k))
        enddo

        allocate(this%loss_cone, mold=loss_cone)
        !$omp parallel default(none) &
        !$omp private(i, k) firstprivate(n_points_RZ, n_points_phi) &
        !$omp shared(this, loss_cone)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ

            if (this%is_core(i, k) == 0.0_GP) then
                ! We define all points outside the closed field line region
                ! to be not in the loss cone
                this%loss_cone(i, k) = 0.0_GP

            else
                this%loss_cone(i, k) = loss_cone(i, k)

            endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel
        deallocate(loss_cone_pol, loss_cone)
    end subroutine
end submodule
