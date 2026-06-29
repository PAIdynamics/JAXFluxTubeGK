submodule (op_rhs_vlasov_eq_static_m) op_rhs_vlasov_eq_static_cpu_s
    use, intrinsic :: iso_fortran_env
    use params_mesh_m, only: get_use_bsg
    ! From PARALLAX
    use csrmat_m, only: csrmat_t

    implicit none

contains

    module subroutine initialize_cpu(this, mesh, bsg_op)
        class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op

        call this%initialize_parent(mesh)
        if (get_use_bsg()) then
            this%bsg_op => bsg_op
        else
            nullify(this%bsg_op)
        end if
    end subroutine

    pure real(kind=GP) function arakawa_stencil(zeta, psi)
        !! Computes Arakawa Bracket [zeta,psi] for given 3x3 arrays zeta and psi
        !! Returns the brackets in the order x,y; x,z; y,z
        real(kind=GP), intent(in), dimension(-1:1,-1:1) :: zeta, psi
        real(kind=GP) :: jplusplus, jpluscross, jcrossplus

        jplusplus = (zeta(1,0) - zeta(-1,0)) * (psi(0,1) - psi(0,-1)) &
                  - (zeta(0,1) - zeta(0,-1)) * (psi(1,0) - psi(-1,0))

        jpluscross = zeta( 1, 0) * (psi( 1, 1) - psi( 1,-1)) &
                   - zeta(-1, 0) * (psi(-1, 1) - psi(-1,-1)) &
                   - zeta( 0, 1) * (psi( 1, 1) - psi(-1, 1)) &
                   + zeta( 0,-1) * (psi( 1,-1) - psi(-1,-1))

        jcrossplus = zeta( 1, 1) * (psi( 0,1) - psi( 1, 0)) &
                   - zeta(-1,-1) * (psi(-1,0) - psi( 0,-1)) &
                   - zeta(-1, 1) * (psi( 0,1) - psi(-1, 0)) &
                   + zeta( 1,-1) * (psi( 1,0) - psi( 0,-1))

        arakawa_stencil = jplusplus + jpluscross + jcrossplus
    end function arakawa_stencil

    module subroutine apply_cpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                da_B_par_in, da_f_out)
        class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in
        class(data_array_2d_t), intent(in) :: da_A_par_in
        class(data_array_2d_t), intent(in) :: da_B_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        if (get_use_bsg()) then
            call this%apply_bsg_cpu(da_f_in, da_phi_in, da_A_par_in, &
                                    da_B_par_in, da_f_out)
        else
            call this%apply_sg_cpu(da_f_in, da_phi_in, da_A_par_in, &
                                   da_B_par_in, da_f_out)
        end if

    end subroutine

    module subroutine apply_sg_cpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                   da_B_par_in, da_f_out)
        class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in
        class(data_array_2d_t), intent(in) :: da_A_par_in
        class(data_array_2d_t), intent(in) :: da_B_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: i, k, l, m, n
        ! Loop indices
        integer :: ox, oz
        ! Loop indices used to fill Arakawa arrays
        integer :: size_phi
        !! Number of poloidal planes without ghost points in the phi dimension
        integer :: k_periodic
        !! Index k for poloidal planes but wrapped around periodically for
        !! ghost points in the phi dimension
        integer :: ngb_m, ngb_p, ngb_mm, ngb_pp, ngb_mp, ngb_pm
        ! Neighbor indices
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        ! Bounds
        integer(kind=INT64) :: ntotal, nn, nm, nl, nk, ni, itotal
        ! Local variables used for manual loop collapsing

        real(kind=GP) :: bps
        ! B parallel star
        real(kind=GP) :: pb
        ! Poisson bracket
        real(kind=GP) :: dfdx, dfdz, dfdy, dfdvp
        ! Derivatives of dist func
        real(kind=GP) :: dphidx, dphidz, dphidy
        ! Derivatives of electrostatic potential
        real(kind=GP) :: dAdx, dAdy, dAdz
        ! Derivatives of electromagnetic vector potential
        real(kind=GP) :: dBpardx, dBpardy, dBpardz
        ! Derivatives of the parallel magnetic fluctuations
        real(kind=GP) :: dH2dx, dH2dy, dH2dz
        ! Derivatives of the second order Hamiltonian
        real(kind=GP) :: f_mm, f_m, f_p, f_pp
        ! Values of dist func used in derivative stencil
        real(kind=GP) :: phi_mm, phi_m, phi_p, phi_pp
        ! Values of phi used in derivative stencil
        real(kind=GP) :: A_par_mm, A_par_m, A_par_p, A_par_pp
        ! Values of Apar used in derivative stencil
        real(kind=GP) :: B_par_mm, B_par_m, B_par_p, B_par_pp
        ! Values of Bpar used in derivative stencil
        real(kind=GP) :: H2_mm, H2_m, H2_p, H2_pp
        ! Values of H2 used in derivative stencil
        real(kind=GP), dimension(-1:1,-1:1) :: zeta, psi
        ! Arrays used in Arakawa stencil
        real(kind=GP) :: grad_H_x, grad_H_y, grad_H_z
        ! Derivatives of the Hamiltonian
        real(kind=GP) :: hyp_xz, hyp_y, hyp_vp, dif_xz
        ! Terms collecting hyperdiffusion and diffusion
        real(kind=GP) :: bs_adv, bx_adv, vp_adv
        ! Right hand side terms

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:):: f_in, f_out
        real(kind=GP), contiguous, pointer, dimension(:,:) :: phi_in, &
                                                              A_par_in, B_par_in
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute, &
            absB, curl_normb_y, dgyzdy_over_g, dgyxdy_over_g, dgyxdz_over_g, &
            dgyzdx_over_g, dabsBdy, dabsBdx, dabsBdz, gyx, gyz, inv_g
        integer, contiguous, pointer, dimension(:,:,:,:) :: neighbors

        type(csrmat_t), dimension(:), pointer :: map_p, map_pp, map_m, map_mm

        lb = da_f_in%get_lbound()
        ub = da_f_in%get_ubound()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in     => da_f_in%get_readonly_pointer()
        phi_in   => da_phi_in%get_readonly_pointer()
        A_par_in => da_A_par_in%get_readonly_pointer()
        B_par_in => da_B_par_in%get_readonly_pointer()
        f_out    => da_f_out%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        ! Get information about the grid and the magnetic equilibrium
        neighbors     => this%mesh%get_neighbor_indices_pointer()
        vp            => this%mesh%get_vp_pointer()
        mu            => this%mesh%get_mu_pointer()
        absB          => this%mesh%get_absB_pointer()
        dabsBdy       => this%mesh%get_dabsBdy_pointer()
        dabsBdx       => this%mesh%get_dabsBdx_pointer()
        dabsBdz       => this%mesh%get_dabsBdz_pointer()
        curl_normb_y  => this%mesh%get_curl_normb_y_pointer()
        gyx           => this%mesh%get_normb_R_pointer()
        gyz           => this%mesh%get_normb_Z_pointer()
        inv_g         => this%mesh%get_inv_g_pointer()
        dgyzdy_over_g => this%mesh%get_dgyzdy_over_g_pointer()
        dgyxdy_over_g => this%mesh%get_dgyxdy_over_g_pointer()
        dgyzdx_over_g => this%mesh%get_dgyzdx_over_g_pointer()
        dgyxdz_over_g => this%mesh%get_dgyxdz_over_g_pointer()

        ! Get map matrices and information about the target
        map_mm     => this%mesh%get_map_negative2_pointer()
        map_m      => this%mesh%get_map_negative1_pointer()
        map_p      => this%mesh%get_map_positive1_pointer()
        map_pp     => this%mesh%get_map_positive2_pointer()
        is_compute => this%mesh%get_is_compute_pointer()

        ! NOTE: This operator uses manual loop collapsing since there is a
        !       bug in the Intel compiler that leads to erroneous collapse
        !       of the OMP loop in release (and debug+coverage) configuration.
        ! TODO: Change this if the bugs have been resolved. In that case check
        !       for performance of manual collapsing of indices since currently
        !       it is also faster than OMP collapse.

        if(.not. allocated(this%H2)) then
            allocate(this%H2(lb(1):ub(1), lb(2):ub(2)))
        endif

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(this, f_in, phi_in, f_out, absB, curl_normb_y, vp, mu, &
        !$omp        neighbors, dgyzdy_over_g, dgyxdy_over_g, dgyzdx_over_g, &
        !$omp        dgyxdz_over_g, dabsBdy, dabsBdx, dabsBdz, gyx, gyz, &
        !$omp        inv_g, map_p, map_m, map_pp, map_mm, &
        !$omp        is_compute, A_par_in, B_par_in) &
        !$omp private(i, k, l, m, n, itotal, ni, nn, nm, nl, nk, ntotal, &
        !$omp         size_phi, k_periodic, ox, oz, psi, zeta, bps, pb, &
        !$omp         ngb_mm, ngb_m, ngb_p, ngb_pp, ngb_pm, ngb_mp,&
        !$omp         dfdz, dfdx, dfdy, dfdvp, dphidx, dphidz, dphidy, &
        !$omp         dAdx, dAdy, dAdz, dBpardx, dBpardy, dBpardz, &
        !$omp         f_mm, f_m, f_p, f_pp, &
        !$omp         phi_mm, phi_m, phi_p, phi_pp, &
        !$omp         A_par_mm, A_par_m, A_par_p, A_par_pp, &
        !$omp         B_par_mm, B_par_m, B_par_p, B_par_pp, &
        !$omp         hyp_xz, hyp_y, hyp_vp, dif_xz, &
        !$omp         grad_H_x, grad_H_y, grad_H_z, bs_adv, bx_adv, vp_adv, &
        !$omp         dH2dx, dH2dy, dH2dz, H2_mm, H2_m, H2_p, H2_pp)

        ni = ub(1) - lb(1) + 1
        nk = ub(2) - lb(2) + 1
        ntotal = nk * ni
        size_phi = this%mesh%size_phi()

        ! The following loop calculates the second order Hamiltonian H2.
        ! As the main loop below calculates derivatives of H2, we calculate
        ! H2 on the ghost points as well.

        !$omp do schedule(static)
        do itotal = 1, ntotal
            i = lb(1) + mod(itotal - 1, ni)
            k = lb(2) + mod(int((itotal - 1) / ni), nk)

            k_periodic = modulo(k - 1, size_phi) + 1

            ngb_m  = neighbors(-1,  0, i, k_periodic)
            ngb_p  = neighbors( 1,  0, i, k_periodic)
            ngb_mm = neighbors( 0, -1, i, k_periodic)
            ngb_pp = neighbors( 0,  1, i, k_periodic)

            ! NOTE: The outermost ghost layer does not have neighbors on the
            !       outside. The neighbors array will contain the index of the
            !       point itself then. Since values of H2 on the outermost
            !       ghost layer are not used, we can just fill it with these
            !       dummy values from the computation with that index.

            dphidx = this%prefac_cd_xz * 1.0_GP / 2.0_GP &
                   * (phi_in(ngb_p,  k) - phi_in(ngb_m,  k))
            dphidz = this%prefac_cd_xz * 1.0_GP / 2.0_GP &
                   * (phi_in(ngb_pp, k) - phi_in(ngb_mm, k))

            ! NOTE: H2 misses a factor of mass in order to avoid a species
            !       dependence of the array and save memory and execution speed
            ! NOTE: We need to use k_periodic here since absB does not contain
            !       values at ghost points in the phi dimension
            this%H2(i, k) = -this%prefac_H2 * (dphidx**2 + dphidz**2) &
                          / (2.0_GP * absB(i, k_periodic)**2)
        enddo
        !$omp enddo

        nn = ub_stripped(5) - lb_stripped(5) + 1
        nm = ub_stripped(4) - lb_stripped(4) + 1
        nl = ub_stripped(3) - lb_stripped(3) + 1
        nk = ub_stripped(2) - lb_stripped(2) + 1
        ni = ub_stripped(1) - lb_stripped(1) + 1
        ntotal = nn * nm * nl * nk * ni
        !$omp do schedule(static)
        do itotal = 1, ntotal
            i = lb_stripped(1) + mod(itotal - 1, ni)
            k = lb_stripped(2) + mod(int((itotal - 1) / ni), nk)
            l = lb_stripped(3) + mod(int((itotal - 1) / (ni*nk)), nl)
            m = lb_stripped(4) + mod(int((itotal - 1) / (ni*nk*nl)), nm)
            n = lb_stripped(5) + mod(int((itotal - 1) / (ni*nk*nl*nm)), nn)

            ! Interpolate on planes k - 2, k - 1, k + 1 and k + 2
            ! to be used for calculating y derivatives
            f_mm = dot_product( &
                   map_mm(k)%val(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                f_in(map_mm(k)%j(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                     k - 2, l, m, n))
            f_m  = dot_product( &
                   map_m(k)%val(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                f_in(map_m(k)%j(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                     k - 1, l, m, n))
            f_p  = dot_product( &
                   map_p(k)%val(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                f_in(map_p(k)%j(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                     k + 1, l, m, n))
            f_pp = dot_product( &
                   map_pp(k)%val(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                f_in(map_pp(k)%j(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                     k + 2, l, m, n))

            phi_mm = dot_product( &
                     map_mm(k)%val(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                phi_in(map_mm(k)%j(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                       k - 2))
            phi_m  = dot_product( &
                     map_m(k)%val(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                phi_in(map_m(k)%j(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                       k - 1))
            phi_p  = dot_product( &
                     map_p(k)%val(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                phi_in(map_p(k)%j(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                       k + 1))
            phi_pp = dot_product( &
                     map_pp(k)%val(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                phi_in(map_pp(k)%j(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                       k + 2))

            A_par_mm = dot_product( &
                       map_mm(k)%val(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                A_par_in(map_mm(k)%j(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                         k - 2))
            A_par_m  = dot_product( &
                       map_m(k)%val(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                A_par_in(map_m(k)%j(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                         k - 1))
            A_par_p  = dot_product( &
                       map_p(k)%val(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                A_par_in(map_p(k)%j(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                         k + 1))
            A_par_pp = dot_product( &
                       map_pp(k)%val(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                A_par_in(map_pp(k)%j(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                         k + 2))

            B_par_mm = dot_product( &
                       map_mm(k)%val(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                B_par_in(map_mm(k)%j(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1), &
                         k - 2))
            B_par_m  = dot_product( &
                       map_m(k)%val(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                B_par_in(map_m(k)%j(map_m(k)%i(i):map_m(k)%i(i + 1) - 1), &
                         k - 1))
            B_par_p  = dot_product( &
                       map_p(k)%val(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                B_par_in(map_p(k)%j(map_p(k)%i(i):map_p(k)%i(i + 1) - 1), &
                         k + 1))
            B_par_pp = dot_product( &
                       map_pp(k)%val(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                B_par_in(map_pp(k)%j(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1), &
                         k + 2))
            ! NOTE: The parallel derivatives of phi and H2 (propto dphi^2)
            !       could be combined to gain some performance.
            ! TODO: Investigate the potential performance gain.

            H2_mm = dot_product( &
                      map_mm(k)%val(map_mm(k)%i(i) : map_mm(k)%i(i + 1) - 1), &
                this%H2(map_mm(k)%j(map_mm(k)%i(i) : map_mm(k)%i(i + 1) - 1), &
                         k - 2))
            H2_m = dot_product( &
                      map_m(k)%val(map_m(k)%i(i) : map_m(k)%i(i + 1) - 1), &
                this%H2(map_m(k)%j(map_m(k)%i(i) : map_m(k)%i(i + 1) - 1), &
                         k - 1))
            H2_p = dot_product( &
                      map_p(k)%val(map_p(k)%i(i) : map_p(k)%i(i + 1) - 1), &
                this%H2(map_p(k)%j(map_p(k)%i(i) : map_p(k)%i(i + 1) - 1), &
                         k + 1))
            H2_pp = dot_product( &
                      map_pp(k)%val(map_pp(k)%i(i) : map_pp(k)%i(i + 1) - 1), &
                this%H2(map_pp(k)%j(map_pp(k)%i(i) : map_pp(k)%i(i + 1) - 1), &
                         k + 2))

            ! Calculate y derivatives

            dfdy = this%prefac_cd_y_mm(i, k) * f_mm &
                 + this%prefac_cd_y_m(i, k)  * f_m &
                 + this%prefac_cd_y_o(i, k)  * f_in(i, k, l, m, n) &
                 + this%prefac_cd_y_p(i, k)  * f_p &
                 + this%prefac_cd_y_pp(i, k) * f_pp

            dphidy = this%prefac_cd_y_mm(i, k) * phi_mm &
                   + this%prefac_cd_y_m(i, k)  * phi_m &
                   + this%prefac_cd_y_o(i, k)  * phi_in(i, k) &
                   + this%prefac_cd_y_p(i, k)  * phi_p &
                   + this%prefac_cd_y_pp(i, k) * phi_pp

            dAdy = this%prefac_cd_y_mm(i, k) * A_par_mm &
                 + this%prefac_cd_y_m(i, k)  * A_par_m &
                 + this%prefac_cd_y_o(i, k)  * A_par_in(i, k) &
                 + this%prefac_cd_y_p(i, k)  * A_par_p &
                 + this%prefac_cd_y_pp(i, k) * A_par_pp

            dBpardy = this%prefac_cd_y_mm(i, k) * B_par_mm &
                    + this%prefac_cd_y_m(i, k)  * B_par_m &
                    + this%prefac_cd_y_o(i, k)  * B_par_in(i, k) &
                    + this%prefac_cd_y_p(i, k)  * B_par_p &
                    + this%prefac_cd_y_pp(i, k) * B_par_pp

            dH2dy = this%prefac_cd_y_mm(i, k) * H2_mm &
                  + this%prefac_cd_y_m(i, k)  * H2_m &
                  + this%prefac_cd_y_o(i, k)  * this%H2(i, k) &
                  + this%prefac_cd_y_p(i, k)  * H2_p &
                  + this%prefac_cd_y_pp(i, k) * H2_pp

            hyp_y =-this%prefac_hyp_y_mm(i, k) * f_mm &
                  - this%prefac_hyp_y_m(i, k)  * f_m &
                  - this%prefac_hyp_y_o(i, k)  * f_in(i, k, l, m, n) &
                  - this%prefac_hyp_y_p(i, k)  * f_p &
                  - this%prefac_hyp_y_pp(i, k) * f_pp

            ! Load the index of neighbors in the x direction
            ngb_mm = neighbors(-2,  0, i, k)
            ngb_m  = neighbors(-1,  0, i, k)
            ngb_mp = neighbors(-1,  1, i, k)
            ngb_pm = neighbors( 1, -1, i, k)
            ngb_p  = neighbors( 1,  0, i, k)
            ngb_pp = neighbors( 2,  0, i, k)

            ! Calculate the parts involving x derivatives for the biharmonic
            ! hyperdiffusion operator
            hyp_xz = -this%prefac_hyp_xz &
                   * (  1.0_GP * f_in(ngb_mm, k, l, m, n) &
                     -  8.0_GP * f_in(ngb_m,  k, l, m, n) &
                     +  2.0_GP * f_in(ngb_mp, k, l, m, n) &
                     + 20.0_GP * f_in(i,      k, l, m, n) &
                     +  2.0_GP * f_in(ngb_pm, k, l, m, n) &
                     -  8.0_GP * f_in(ngb_p,  k, l, m, n) &
                     +  1.0_GP * f_in(ngb_pp, k, l, m, n))

            ! Calculate the parts involving x derivatives for the harmonic
            ! diffusion operator applied in the buffer zones
            dif_xz = this%prefac_buffer_zone(i, k) &
                   * ( 1.0_GP * f_in(ngb_m, k, l, m, n) &
                     - 4.0_GP * f_in(i,     k, l, m, n) &
                     + 1.0_GP * f_in(ngb_p, k, l, m, n))

            ! Calculate x derivatives
            dfdx = this%prefac_cd_xz &
                 * (- 1.0_GP / 12.0_GP * f_in(ngb_pp, k, l, m, n) &
                    + 2.0_GP /  3.0_GP * f_in(ngb_p,  k, l, m, n) &
                    - 2.0_GP /  3.0_GP * f_in(ngb_m,  k, l, m, n) &
                    + 1.0_GP / 12.0_GP * f_in(ngb_mm, k, l, m, n))

            dphidx = this%prefac_cd_xz &
                   * (- 1.0_GP / 12.0_GP * phi_in(ngb_pp, k) &
                      + 2.0_GP /  3.0_GP * phi_in(ngb_p,  k) &
                      - 2.0_GP /  3.0_GP * phi_in(ngb_m,  k) &
                      + 1.0_GP / 12.0_GP * phi_in(ngb_mm, k))

            dAdx = this%prefac_cd_xz &
                 * (- 1.0_GP / 12.0_GP * A_par_in(ngb_pp, k) &
                    + 2.0_GP /  3.0_GP * A_par_in(ngb_p,  k) &
                    - 2.0_GP /  3.0_GP * A_par_in(ngb_m,  k) &
                    + 1.0_GP / 12.0_GP * A_par_in(ngb_mm, k))

            dBpardx = this%prefac_cd_xz &
                    * (- 1.0_GP / 12.0_GP * B_par_in(ngb_pp, k) &
                       + 2.0_GP /  3.0_GP * B_par_in(ngb_p,  k) &
                       - 2.0_GP /  3.0_GP * B_par_in(ngb_m,  k) &
                       + 1.0_GP / 12.0_GP * B_par_in(ngb_mm, k))

            dH2dx = this%prefac_cd_xz &
                  * (- 1.0_GP / 12.0_GP * this%H2(ngb_pp, k) &
                     + 2.0_GP /  3.0_GP * this%H2(ngb_p,  k) &
                     - 2.0_GP /  3.0_GP * this%H2(ngb_m,  k) &
                     + 1.0_GP / 12.0_GP * this%H2(ngb_mm, k))

            ! Load the index of neighbors in the z direction
            ngb_mm = neighbors( 0, -2, i, k)
            ngb_m  = neighbors( 0, -1, i, k)
            ngb_mp = neighbors(-1, -1, i, k)
            ngb_pm = neighbors( 1,  1, i, k)
            ngb_p  = neighbors( 0,  1, i, k)
            ngb_pp = neighbors( 0,  2, i, k)

            ! Calculate the parts involving z derivatives for the biharmonic
            ! hyperdiffusion operator
            hyp_xz = hyp_xz - this%prefac_hyp_xz &
                   * ( 1.0_GP * f_in(ngb_mm, k, l, m, n) &
                     - 8.0_GP * f_in(ngb_m,  k, l, m, n) &
                     + 2.0_GP * f_in(ngb_mp, k, l, m, n) &
                     + 2.0_GP * f_in(ngb_pm, k, l, m, n) &
                     - 8.0_GP * f_in(ngb_p,  k, l, m, n) &
                     + 1.0_GP * f_in(ngb_pp, k, l, m, n))

            ! Calculate the parts involving z derivatives for the harmonic
            ! diffusion operator applied in the buffer zones
            dif_xz = dif_xz + this%prefac_buffer_zone(i, k) &
                   * (f_in(ngb_m, k, l, m, n) + f_in(ngb_p, k, l, m, n))

            ! Calculate z derivatives
            dfdz = this%prefac_cd_xz &
                 * (- 1.0_GP / 12.0_GP * f_in(ngb_pp, k, l, m, n) &
                    + 2.0_GP /  3.0_GP * f_in(ngb_p,  k, l, m, n) &
                    - 2.0_GP /  3.0_GP * f_in(ngb_m,  k, l, m, n) &
                    + 1.0_GP / 12.0_GP * f_in(ngb_mm, k, l, m, n))

            dphidz = this%prefac_cd_xz &
                   * (- 1.0_GP / 12.0_GP * phi_in(ngb_pp, k) &
                      + 2.0_GP /  3.0_GP * phi_in(ngb_p,  k) &
                      - 2.0_GP /  3.0_GP * phi_in(ngb_m,  k) &
                      + 1.0_GP / 12.0_GP * phi_in(ngb_mm, k))

            dAdz = this%prefac_cd_xz &
                 * (- 1.0_GP / 12.0_GP * A_par_in(ngb_pp, k) &
                    + 2.0_GP /  3.0_GP * A_par_in(ngb_p,  k) &
                    - 2.0_GP /  3.0_GP * A_par_in(ngb_m,  k) &
                    + 1.0_GP / 12.0_GP * A_par_in(ngb_mm, k))

            dBpardz = this%prefac_cd_xz &
                    * (- 1.0_GP / 12.0_GP * B_par_in(ngb_pp, k) &
                       + 2.0_GP /  3.0_GP * B_par_in(ngb_p,  k) &
                       - 2.0_GP /  3.0_GP * B_par_in(ngb_m,  k) &
                       + 1.0_GP / 12.0_GP * B_par_in(ngb_mm, k))

            dH2dz = this%prefac_cd_xz &
                  * (- 1.0_GP / 12.0_GP * this%H2(ngb_pp, k) &
                     + 2.0_GP /  3.0_GP * this%H2(ngb_p,  k) &
                     - 2.0_GP /  3.0_GP * this%H2(ngb_m,  k) &
                     + 1.0_GP / 12.0_GP * this%H2(ngb_mm, k))

            ! Calculate vp derivatives
            dfdvp = this%prefac_cd_vp &
                  * (- 1.0_GP / 12.0_GP * f_in(i, k, l + 2, m, n) &
                     + 2.0_GP /  3.0_GP * f_in(i, k, l + 1, m, n) &
                     - 2.0_GP /  3.0_GP * f_in(i, k, l - 1, m, n) &
                     + 1.0_GP / 12.0_GP * f_in(i, k, l - 2, m, n))

            hyp_vp = -this%prefac_hyp_vp &
                   * ( 1.0_GP * f_in(i, k, l - 2, m, n) &
                     - 4.0_GP * f_in(i, k, l - 1, m, n) &
                     + 6.0_GP * f_in(i, k, l    , m, n) &
                     - 4.0_GP * f_in(i, k, l + 1, m, n) &
                     + 1.0_GP * f_in(i, k, l + 2, m, n))

            ! Calculate the right hand side

            bps = absB(i, k) + this%prefac_bps(n) * vp(l) * curl_normb_y(i, k)

            ! Fill zeta and psi for the calculation of the Poisson brackets
            ! {q*phi, f}_zx, {T*mu*B_par, f}_zx, {sqrt(2/m)*q*vp*A_par, f}_zx
            ! with the Arakawa method.
            !
            ! NOTE: The function arakawa_stencil calculates the Poisson
            !       brackets in the order x;z. Therefore a minus is added to
            !       the arguments.
            ! A.12 (a) 4th line (flutter terms)
            ! A.13 (b) 4th line
            do oz = -1, 1
            do ox = -1, 1
                ngb_m = neighbors(ox, oz, i, k)
                zeta(ox, oz) = &
                    + this%charges(n)            * phi_in(ngb_m, k) &
                    + this%prefac_Bpar(n) * mu(m) * B_par_in(ngb_m, k) &
                    - this%prefac_flutter_xyz(n) * vp(l) &
                                                 * A_par_in(ngb_m, k) &
                    + this%masses(n)             * this%H2(ngb_m, k)

                psi(ox, oz) = f_in(ngb_m, k, l, m, n)
            enddo
            enddo

            grad_H_x = this%temp_scalings(n) * mu(m) * dabsBdx(i, k) &
                     + this%charges(n) * dphidx &
                     + this%masses(n) * dH2dx &
                     + this%prefac_Bpar(n) * mu(m) * dBpardx
            grad_H_y = this%temp_scalings(n) * mu(m) * dabsBdy(i, k) &
                     + this%charges(n) * dphidy &
                     + this%masses(n) * dH2dy &
                     + this%prefac_Bpar(n) * mu(m) * dBpardy
            grad_H_z = this%temp_scalings(n) * mu(m) * dabsBdz(i, k) &
                     + this%charges(n) * dphidz &
                     + this%masses(n) * dH2dz &
                     + this%prefac_Bpar(n) * mu(m) * dBpardz

            ! 1) Calculate cross field advection

            ! These terms can be expressed as
            ! Poisson brackets. Calculate the Poisson brackets
            ! {q*phi, f}_zx, {T*mu*B_par, f}_zx, {sqrt(2/m)*q*vp*A_par, f}_zx
            ! with the Arakawa method
            pb = this%prefac_arakawa * arakawa_stencil(zeta, psi)

            ! Calculate the remaining Poisson brackets
            ! {muB, f}_zx, bR{muB, f}_yz, bZ{muB, f}_xy
            ! and
            ! {q*phi, f}_yz, {q*phi, f}_xy
            ! and
            ! {T*mu*B_par, f}_yz, {T*mu*B_par, f}_xy
            ! and
            ! {sqrt(2/m)*q*vp A_par, f}_yz, {sqrt(2/m)*q*vp*A_par, f}_xy
            ! with centered finite differences
            ! A.12 (a) 3rd line (flutter terms)
            ! A.13 (b) 2nd line -> mu*dB
            pb = pb &
               - dfdz * (-mu(m) * this%temp_scalings(n) * dabsBdx(i, k) &
                         + gyx(i, k) &
                            * (grad_H_y &
                                - this%prefac_flutter_xyz(n) * vp(l) * dAdy)) &
               - dfdx * ( mu(m) * this%temp_scalings(n) * dabsBdz(i, k) &
                         - gyz(i, k) &
                            * (grad_H_y &
                                - this%prefac_flutter_xyz(n) * vp(l) * dAdy)) &
               - dfdy * (- gyx(i, k) &
                            * (grad_H_z &
                                - this%prefac_flutter_xyz(n) * vp(l) * dAdz) &
                         + gyz(i, k) &
                            * (grad_H_x &
                                - this%prefac_flutter_xyz(n) * vp(l) * dAdx))

            bx_adv = pb * inv_g(i, k) * this%prefac_orth(n)

            ! 2) Calculate advection along corrected magnetic field B^*

            ! Contribution of the parallel advection
            ! A.12 (a) 1st line
            bs_adv = -2.0_GP * this%prefac_par(n)* vp(l) * absB(i, k) * dfdy
            ! Contribution of the curvature term
            ! A.12 (a) 2nd line
            bs_adv = bs_adv + 2.0_GP * vp(l) * this%prefac_orth(n) * vp(l) &
                   * (                      - dgyzdy_over_g(i, k)  * dfdx &
                     + (dgyzdx_over_g(i, k) - dgyxdz_over_g(i, k)) * dfdy &
                     +  dgyxdy_over_g(i, k)                        * dfdz)
            bs_adv = bs_adv * this%temp_scalings(n)

            ! 3) Calculate advection in velocity space

            ! Contribution of the parallel advection
            ! A.14 (c) 1st+6th line very first terms
            vp_adv = this%prefac_par(n) * absB(i, k) * grad_H_y * dfdvp
            ! Contribution of the curvature term
            ! A.14 (c) 2nd line + 6th line second part + 7th line
            vp_adv = vp_adv - this%prefac_orth(n) * vp(l) &
                   * (                      - dgyzdy_over_g(i, k)  * grad_H_x &
                     + (dgyzdx_over_g(i, k) - dgyxdz_over_g(i, k)) * grad_H_y &
                     +  dgyxdy_over_g(i, k)                        * grad_H_z) &
                   * dfdvp
            ! Contribution of the electromagnetic flutter
            ! A.14 (c) 3-5th + 8-10th line
            vp_adv = vp_adv - this%prefac_flutter_vp(n) * inv_g(i, k) &
                   * ( grad_H_x * (            dAdz - gyz(i, k) * dAdy) &
                     + grad_H_z * (gyx(i, k) * dAdy -             dAdx) &
                     + grad_H_y * (gyz(i, k) * dAdx - gyx(i, k) * dAdz)) &
                   * dfdvp

            ! Collect all terms

            f_out(i, k, l, m, n) = (bs_adv + bx_adv + vp_adv) / bps &
                                 + hyp_xz + hyp_vp + hyp_y + dif_xz

            f_out(i, k, l, m, n) = f_out(i, k, l, m, n) * is_compute(i, k)
        enddo
        !$omp end do nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

end submodule
