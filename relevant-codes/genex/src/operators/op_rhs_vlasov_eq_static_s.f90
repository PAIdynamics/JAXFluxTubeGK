submodule (op_rhs_vlasov_eq_static_m) op_rhs_vlasov_eq_static_s
    use, intrinsic :: iso_fortran_env
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_mesh_m, only: get_n_points_sp
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_numerical_scheme_m, only: get_hyp_xz, get_hyp_vp, get_hyp_y, &
                                         get_buf_zone_strength, &
                                         get_buf_zone_strength_axis
    use params_gyrokinetic_system_m, only: get_with_em_flutter, &
                                           get_with_nlin_polarization, &
                                           get_with_bpar
    use lagrange_interpolator_m, only: monomials_first_derivative, &
                                       monomials_fourth_derivative
    use mesh_5d_m, only: BUF_ZONE_BOUNDARY_IN, BUF_ZONE_BOUNDARY_OUT, &
                         BUF_ZONE_AXIS, BUF_ZONE_PARCON
    implicit none

contains

    module subroutine initialize_parent_base(this, mesh)
        class(op_rhs_vlasov_eq_static_base_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        integer :: i, k, n, n_points_RZ, n_points_phi, n_points_sp
        real(kind=GP), pointer, dimension(:,:) :: y_mm, y_m, y_p, y_pp, &
                                                  is_compute
        real(kind=GP) :: monomial_mm, monomial_m, monomial_o, &
                         monomial_p, monomial_pp
        integer, contiguous, pointer, dimension(:,:) :: buf_zone

        this%mesh => mesh

        ! Initialize prefactors for the parallel derivative
        n_points_RZ  = mesh%size_RZ()
        n_points_phi = mesh%size_phi()

        is_compute => mesh%get_is_compute_pointer()

        y_p  => mesh%get_fll_positive1_pointer()
        y_pp => mesh%get_fll_positive2_pointer()
        y_m  => mesh%get_fll_negative1_pointer()
        y_mm => mesh%get_fll_negative2_pointer()

        allocate(this%prefac_cd_y_mm(n_points_RZ, n_points_phi))
        allocate(this%prefac_cd_y_m,   mold=this%prefac_cd_y_mm)
        allocate(this%prefac_cd_y_o,   mold=this%prefac_cd_y_mm)
        allocate(this%prefac_cd_y_p,   mold=this%prefac_cd_y_mm)
        allocate(this%prefac_cd_y_pp,  mold=this%prefac_cd_y_mm)

        allocate(this%prefac_hyp_y_mm, mold=this%prefac_cd_y_mm)
        allocate(this%prefac_hyp_y_m,  mold=this%prefac_cd_y_mm)
        allocate(this%prefac_hyp_y_o,  mold=this%prefac_cd_y_mm)
        allocate(this%prefac_hyp_y_p,  mold=this%prefac_cd_y_mm)
        allocate(this%prefac_hyp_y_pp, mold=this%prefac_cd_y_mm)

        !$omp parallel default(none) &
        !$omp firstprivate(n_points_RZ, n_points_phi) &
        !$omp private(i, k, monomial_mm, monomial_m, monomial_o, &
        !$omp         monomial_p, monomial_pp) &
        !$omp shared(this, is_compute, y_mm, y_m, y_p, y_pp)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
        if(is_compute(i, k) == 1.0_GP) then
            ! NOTE: The field line length is always positive. Hence a minus
            !       sign has to be added for the y grid values going in
            !       opposite direction of the magnetic field line.
            call monomials_first_derivative( &
                -y_mm(i, k), -y_m(i, k), 0.0_GP, y_p(i, k), y_pp(i, k), &
                monomial_mm, monomial_m, monomial_o, monomial_p, monomial_pp)

            this%prefac_cd_y_mm(i, k)  = monomial_mm
            this%prefac_cd_y_m(i, k)   = monomial_m
            this%prefac_cd_y_o(i, k)   = monomial_o
            this%prefac_cd_y_p(i, k)   = monomial_p
            this%prefac_cd_y_pp(i, k)  = monomial_pp

            call monomials_fourth_derivative( &
                -y_mm(i, k), -y_m(i, k), 0.0_GP, y_p(i, k), y_pp(i, k), &
                monomial_mm, monomial_m, monomial_o, monomial_p, monomial_pp)

            this%prefac_hyp_y_mm(i, k) = get_hyp_y() * monomial_mm
            this%prefac_hyp_y_m(i, k)  = get_hyp_y() * monomial_m
            this%prefac_hyp_y_o(i, k)  = get_hyp_y() * monomial_o
            this%prefac_hyp_y_p(i, k)  = get_hyp_y() * monomial_p
            this%prefac_hyp_y_pp(i, k) = get_hyp_y() * monomial_pp
        else
            ! Set the prefactors on the non-compute points to zero
            this%prefac_cd_y_mm(i, k)  = 0.0_GP
            this%prefac_cd_y_m(i, k)   = 0.0_GP
            this%prefac_cd_y_o(i, k)   = 0.0_GP
            this%prefac_cd_y_p(i, k)   = 0.0_GP
            this%prefac_cd_y_pp(i, k)  = 0.0_GP

            this%prefac_hyp_y_mm(i, k) = 0.0_GP
            this%prefac_hyp_y_m(i, k)  = 0.0_GP
            this%prefac_hyp_y_o(i, k)  = 0.0_GP
            this%prefac_hyp_y_p(i, k)  = 0.0_GP
            this%prefac_hyp_y_pp(i, k) = 0.0_GP
        endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        ! Initialize prefactors for the x, z derivatives
        this%prefac_arakawa = 1.0_GP / (12.0_GP * mesh%delta_RZ()**2)
        this%prefac_cd_xz   = 1.0_GP / mesh%delta_RZ()
        this%prefac_hyp_xz  = get_hyp_xz()

        ! Initialize buffer zone prefactors
        buf_zone => mesh%get_buf_zone_pointer()

        allocate(this%prefac_buffer_zone(n_points_RZ, n_points_phi))
        !$omp parallel default(none) private(i, k) &
        !$omp firstprivate(n_points_RZ, n_points_phi) &
        !$omp shared(this, buf_zone)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            if(buf_zone(i, k) == BUF_ZONE_BOUNDARY_IN .or. &
               buf_zone(i, k) == BUF_ZONE_BOUNDARY_OUT .or. &
               buf_zone(i, k) == BUF_ZONE_PARCON) then
                this%prefac_buffer_zone(i, k) = get_buf_zone_strength()
            elseif(buf_zone(i, k) == BUF_ZONE_AXIS) then
                this%prefac_buffer_zone(i, k) = get_buf_zone_strength_axis()
            else
                this%prefac_buffer_zone(i, k) = 0.0_GP
            endif
        enddo
        !$omp end do
        enddo
        !$omp end parallel

        n_points_sp = get_n_points_sp()
        allocate(this%charges(n_points_sp))
        allocate(this%masses(n_points_sp))
        allocate(this%temp_scalings(n_points_sp))
        allocate(this%prefac_Bpar(n_points_sp))

        do n = 1, n_points_sp
            this%masses(n) = get_mass(n)
            this%charges(n) = get_charge(n)
            this%temp_scalings(n) = get_temp_scaling(n)
        enddo

        ! Initialize prefactors for the physical terms
        if(get_with_nlin_polarization()) then
            this%prefac_H2 = (get_rho_ref() / get_L_ref())**2
        else
            this%prefac_H2 = 0.0_GP
        endif

        if(get_with_bpar()) then
            this%prefac_Bpar = this%temp_scalings
        else
            this%prefac_Bpar = 0.0_GP
        endif
    end subroutine

    module subroutine initialize_parent(this, mesh)
        class(op_rhs_vlasov_eq_static_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        integer :: n, n_points_sp

        call this%initialize_parent_base(mesh)

        this%n_flops = 677   ! We estimate division with 4
        this%n_ls    = 266

        ! Initialize prefactors for the vp derivatives
        this%prefac_cd_vp   = 1.0_GP / mesh%delta_vp()
        this%prefac_hyp_vp  = get_hyp_vp()

        n_points_sp = get_n_points_sp()
        allocate(this%prefac_bps(n_points_sp))
        allocate(this%prefac_orth(n_points_sp))
        allocate(this%prefac_par(n_points_sp))
        allocate(this%prefac_flutter_xyz(n_points_sp))
        allocate(this%prefac_flutter_vp(n_points_sp))
        do n = 1, n_points_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                             * get_temp_scaling(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
            this%prefac_orth(n) = get_rho_ref() / (get_L_ref() * get_charge(n))
            this%prefac_par(n) = 1.0_GP / sqrt(2.0_GP * this%masses(n) &
                                                      * get_temp_scaling(n))
            if(get_with_em_flutter()) then
                this%prefac_flutter_xyz(n) = sqrt(2.0_GP * get_temp_scaling(n) &
                                                         / this%masses(n)) &
                                           * this%charges(n)
                this%prefac_flutter_vp(n) = get_rho_ref() / get_L_ref() &
                                          / sqrt(2.0_GP * this%masses(n) &
                                                        * get_temp_scaling(n))
            else
                this%prefac_flutter_xyz(n) = 0.0_GP
                this%prefac_flutter_vp(n) = 0.0_GP
            endif
        enddo

    end subroutine

end submodule
