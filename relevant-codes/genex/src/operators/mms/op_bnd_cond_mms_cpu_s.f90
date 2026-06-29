submodule(op_bnd_cond_m) op_bnd_cond_mms_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI, krond
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_q, get_nrad, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_species_m, only: get_is_electrons, get_mass
    use params_mesh_m, only: get_use_vspectral, get_use_bsg
    use params_bsg_m, only: get_num_bsg_blocks
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use array_generation_m, only: linspace
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_mms_cpu(this, dcomm_handler, mesh)
        class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: n

        this%mesh => mesh
        this%R    => this%mesh%get_R_pointer()
        this%Z    => this%mesh%get_Z_pointer()
        this%phi  => this%mesh%get_phi_pointer()
        this%vp   => this%mesh%get_vp_pointer()
        this%mu   => this%mesh%get_mu_pointer()

        do n = 1, mesh%size_sp()
            if(get_is_electrons(n)) then
                this%prefac_co_qn_electrons = (get_rho_ref() / get_L_ref())**2 &
                                            * get_mass(n)
            else
                this%prefac_co_qn_ions = (get_rho_ref() / get_L_ref())**2 &
                                       * get_mass(n)
            end if
        end do

    end subroutine

    module subroutine apply_mms_cpu(this, da_f_inout, da_co_qn_eq, &
                                    da_b_qn_eq, da_b_amps_law, da_b_ohms_law, t)
        class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        this%n_iterations = da_f_inout%get_size()
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call bnd_cond_mms_vspec(this, da_f_inout, da_co_qn_eq, &
                                    da_b_qn_eq, da_b_amps_law, &
                                    da_b_ohms_law, t)
        else
            call bnd_cond_mms(this, da_f_inout, da_co_qn_eq, &
                              da_b_qn_eq, da_b_amps_law, &
                              da_b_ohms_law, t)
        endif

        call this%perf_counter%end_measurement()

    end subroutine

    subroutine bnd_cond_mms(this, da_f_inout, da_co_qn_eq, &
                            da_b_qn_eq, da_b_amps_law, &
                            da_b_ohms_law, t)
        class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        logical :: is_electrons, bnd_to_set
        integer :: i, k, l, m, n, npol, ntor, nrad, size_vp
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), contiguous, pointer, dimension(:,:) :: co_qn_eq, &
                                                              b_qn_eq, &
                                                              b_amps_law, &
                                                              b_ohms_law
        real(kind=GP) :: omega, R, phi, Z, vp, mu, Lref, rhoref, q, &
                         rhomin, rhomax, minorr, shear, ellax1, ellax2
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons, &
                         mms_solution_es_pot, mms_solution_a_par, &
                         mms_solution_e_par, mms_density_ions, &
                         mms_density_electrons
        real(kind=GP), contiguous, pointer, dimension(:, :) :: not_ghost
        integer, contiguous, pointer, dimension(:, :) :: not_in_target
        real(kind=GP), contiguous, pointer, dimension(:, :) :: jacobian, absB
        integer :: nb_point
        logical :: use_bsg
        integer, contiguous, pointer, dimension(:) :: bsg_flags

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout    => da_f_inout%get_pointer()
        co_qn_eq   => da_co_qn_eq%get_pointer()
        b_qn_eq    => da_b_qn_eq%get_pointer()
        b_amps_law => da_b_amps_law%get_pointer()
        b_ohms_law => da_b_ohms_law%get_pointer()

        ! Set MMS parameters
        rhoref = get_rho_ref()
        Lref   = get_L_ref()
        omega  = get_omega()
        q      = get_q()
        npol   = get_npol()
        ntor   = get_ntor()
        nrad   = get_nrad()
        minorr = get_minor_r()
        shear  = get_shear()
        ellax1 = get_ell_ax1()
        ellax2 = get_ell_ax2()
        rhomin = get_rho_min()
        rhomax = get_rho_max()

        not_ghost => this%mesh%get_not_ghost_pointer()
        not_in_target => this%mesh%get_not_in_target_pointer()
        jacobian  => this%mesh%get_jacobian_pointer()
        absB => this%mesh%get_absB_pointer()

        size_vp = this%mesh%size_vp()
        use_bsg = get_use_bsg()
        bsg_flags => this%mesh%get_bsg_flags_pointer()

        ! On first time run allocate and initialize the total vp array
        if(.not. allocated(this%vp_total)) then
            ! Create vp array with ghost cells to set
            ! the MMS solution there
            block
                real(kind=GP) :: length
                integer :: n_ghost_vp, n_total_vp
                integer :: num_blocks

                num_blocks = get_num_bsg_blocks()
                if (.not. get_use_bsg()) num_blocks = 1

                ! Ghosts on one side of vp array
                n_ghost_vp = lb_stripped(3) - lb(3)
                ! Number of points of new, total vp array with ghosts
                n_total_vp  = size_vp + 2 * n_ghost_vp
                ! Length of total vp array with ghosts
                ! NOTE: Due to symmetric 2-sided vp the absolute
                !       value at the lower and higher stripped
                !       boundaries are the same and we only specify
                !       the 1-sided length here.
                allocate(this%vp_total(num_blocks, (1 - n_ghost_vp): &
                                                   (size_vp + n_ghost_vp)))

                do nb_point = 1, num_blocks
                    this%vp   => this%mesh%get_vp_pointer(nb_point)
                    length = -this%vp(1) &
                           + n_ghost_vp * this%mesh%delta_vp(nb_point)
                    this%vp_total(nb_point, :) = linspace(-length, length, &
                                                          n_total_vp, &
                                                          endpoint=.true.)
                end do
            end block
        end if

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, &
        !$omp              t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, not_ghost, not_in_target, co_qn_eq, b_qn_eq, &
        !$omp        b_amps_law, b_ohms_law, jacobian, absB) &
        !$omp private(k, i, mms_solution_es_pot, mms_solution_a_par, &
        !$omp         mms_solution_e_par, R, Z, phi, bnd_to_set, &
        !$omp         mms_density_ions, mms_density_electrons)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ! Set the potentials to the exact solution on the
            ! boundaries and in the divertor target
            bnd_to_set = (not_ghost(i, k) == 0.0_GP &
                          .or. not_in_target(i, k) == 0)
            if(.not. bnd_to_set) cycle

            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)

            if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_es_pot.txt"
include "../../mms/slab/mms_solution_a_par.txt"
include "../../mms/slab/mms_solution_e_par.txt"
include "../../mms/slab/mms_density_ions.txt"
include "../../mms/slab/mms_density_electrons.txt"
            else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_es_pot.txt"
include "../../mms/circular/mms_solution_a_par.txt"
include "../../mms/circular/mms_solution_e_par.txt"
include "../../mms/circular/mms_density_ions.txt"
include "../../mms/circular/mms_density_electrons.txt"
            else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_es_pot.txt"
include "../../mms/dommaschk/mms_solution_a_par.txt"
include "../../mms/dommaschk/mms_solution_e_par.txt"
include "../../mms/dommaschk/mms_density_ions.txt"
include "../../mms/dommaschk/mms_density_electrons.txt"
            else
include "../../mms/salpha/mms_solution_es_pot.txt"
include "../../mms/salpha/mms_solution_a_par.txt"
include "../../mms/salpha/mms_solution_e_par.txt"
include "../../mms/salpha/mms_density_ions.txt"
include "../../mms/salpha/mms_density_electrons.txt"
            end if

            b_qn_eq(i, k)    = mms_solution_es_pot
            b_amps_law(i, k) = mms_solution_a_par
            b_ohms_law(i, k) = mms_solution_e_par

            co_qn_eq(i, k) = &
                (this%prefac_co_qn_ions * mms_density_ions &
                + this%prefac_co_qn_electrons * mms_density_electrons) &
                * jacobian(i, k) / absB(i, k)**2

            end do
        !$omp end do nowait
        end do
        !$omp end parallel

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, &
        !$omp              t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(f_inout, this, not_ghost, not_in_target, size_vp, &
        !$omp        bsg_flags) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, is_electrons, &
        !$omp         mms_solution_f_electrons, &
        !$omp         mms_solution_f_ions, bnd_to_set, nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb(3), ub(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ! Set the distribution function to the exact solution
            ! on the real space and vp boundaries, and in the
            ! divertor target
            bnd_to_set = (not_ghost(i, k) == 0.0_GP &
                          .or. not_in_target(i, k) == 0 &
                          .or. l < 1 &
                          .or. l > size_vp)
            if(.not. bnd_to_set) cycle

            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            nb_point = unpack_bflag(bsg_flags(i))
            vp  = this%vp_total(nb_point, l)
            mu  = this%mu(m)

            if(is_electrons) then
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_electrons.txt"
                else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_f_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_electrons.txt"
                end if
                f_inout(i, k, l, m, n) = mms_solution_f_electrons
            else
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_ions.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_ions.txt"
                else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_f_ions.txt"
                else
include "../../mms/salpha/mms_solution_f_ions.txt"
                end if
                f_inout(i, k, l, m, n) = mms_solution_f_ions
            end if
        end do
        !$omp end do nowait
        end do
        end do
        end do
        end do
        !$omp end parallel

    end subroutine

    subroutine bnd_cond_mms_vspec(this, da_f_inout, da_co_qn_eq, &
                                  da_b_qn_eq, da_b_amps_law, &
                                  da_b_ohms_law, t)
        class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        logical :: is_electrons, bnd_to_set
        integer :: i, k, l, m, n, size_vp, npol, ntor, nrad
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), contiguous, pointer, dimension(:,:) :: co_qn_eq, &
                                                              b_qn_eq, &
                                                              b_amps_law, &
                                                              b_ohms_law
        real(kind=GP) :: omega, R, phi, Z, vp, mu, Lref, rhoref, q, &
                         rhomin, rhomax, minorr, shear, &
                         absB, dabsBdR, &
                         dabsBdphi, dabsBdZ, &
                         bR, bphi, bZ, dbRdphi, &
                         dbRdZ, dbphidR, dbphidZ, &
                         dbZdR, dbZdphi
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons, &
                         mms_solution_es_pot, mms_solution_a_par, &
                         mms_solution_e_par, mms_density_ions, &
                         mms_density_electrons
        real(kind=GP), contiguous, pointer, dimension(:, :) :: not_ghost
        integer, contiguous, pointer, dimension(:, :) :: not_in_target
        real(kind=GP), contiguous, pointer, dimension(:, :) :: jacobian

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout    => da_f_inout%get_pointer()
        co_qn_eq   => da_co_qn_eq%get_pointer()
        b_qn_eq    => da_b_qn_eq%get_pointer()
        b_amps_law => da_b_amps_law%get_pointer()
        b_ohms_law => da_b_ohms_law%get_pointer()

        ! Set MMS parameters
        rhoref = get_rho_ref()
        Lref   = get_L_ref()
        omega  = get_omega()
        q      = get_q()
        npol   = get_npol()
        ntor   = get_ntor()
        nrad   = get_nrad()
        minorr = get_minor_r()
        shear  = get_shear()
        rhomin = this%mesh%rho_min()
        rhomax = this%mesh%rho_max()

        not_ghost => this%mesh%get_not_ghost_pointer()
        not_in_target => this%mesh%get_not_in_target_pointer()
        jacobian  => this%mesh%get_jacobian_pointer()
        size_vp = this%mesh%size_vp()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, &
        !$omp              t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear) &
        !$omp shared(this, not_ghost, not_in_target, co_qn_eq, b_qn_eq, &
        !$omp        b_amps_law, b_ohms_law, jacobian) &
        !$omp private(k, i, mms_solution_es_pot, mms_solution_a_par, &
        !$omp         mms_solution_e_par, R, Z, phi, bnd_to_set, &
        !$omp         mms_density_ions, mms_density_electrons, &
        !$omp         dabsBdR, dabsBdphi, dabsBdZ, bR, bphi, bZ, &
        !$omp         dbRdphi, dbRdZ, dbphidR, dbphidZ, dbZdR, &
        !$omp         absB, dbZdphi)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ! Set the potentials to the exact solution on the
            ! boundaries and in the divertor target
            bnd_to_set = (not_ghost(i, k) == 0.0_GP &
                          .or. not_in_target(i, k) == 0)
            if(.not. bnd_to_set) cycle

            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)

            if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_magfield.txt"
include "../../mms/slab/mms_solution_es_pot.txt"
include "../../mms/slab/mms_solution_a_par.txt"
include "../../mms/slab/mms_solution_e_par.txt"
include "../../mms/slab/mms_density_vspec_ions.txt"
include "../../mms/slab/mms_density_vspec_electrons.txt"
            else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_magfield.txt"
include "../../mms/circular/mms_solution_es_pot.txt"
include "../../mms/circular/mms_solution_a_par.txt"
include "../../mms/circular/mms_solution_e_par.txt"
include "../../mms/circular/mms_density_vspec_ions.txt"
include "../../mms/circular/mms_density_vspec_electrons.txt"
            else
include "../../mms/salpha/mms_magfield.txt"
include "../../mms/salpha/mms_solution_es_pot.txt"
include "../../mms/salpha/mms_solution_a_par.txt"
include "../../mms/salpha/mms_solution_e_par.txt"
include "../../mms/salpha/mms_density_vspec_ions.txt"
include "../../mms/salpha/mms_density_vspec_electrons.txt"
            end if

            b_qn_eq(i, k)    = mms_solution_es_pot
            b_amps_law(i, k) = mms_solution_a_par
            b_ohms_law(i, k) = mms_solution_e_par

             co_qn_eq(i, k) = &
                (this%prefac_co_qn_ions * mms_density_ions &
                + this%prefac_co_qn_electrons * mms_density_electrons) &
                * jacobian(i, k) / absB**2

        end do
        !$omp end do nowait
        end do
        !$omp end parallel

        ! Set boundary conditions to the dist func
        ! for the spectral approach

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, &
        !$omp              t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, &
        !$omp              rhomin, &
        !$omp              rhomax, minorr, shear) &
        !$omp shared(f_inout, this, not_ghost, not_in_target) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, is_electrons, &
        !$omp         mms_solution_f_electrons, mms_solution_f_ions, &
        !$omp         bnd_to_set)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)

            ! NOTE: The mms solution of the dist func is set
            !       only the real boundary space and in
            !       the divertor target. The spectral dist func
            !       is zero on the vp and mu boundary at all time.

            bnd_to_set = (not_ghost(i, k) == 0.0_GP &
                          .or. not_in_target(i, k) == 0)
            if(.not. bnd_to_set) cycle

            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            vp  = this%vp(l)
            mu  = this%mu(m)

            if(is_electrons) then
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_electrons.txt"
                end if
                    f_inout(i, k, l, m, n) = mms_solution_f_electrons
            else
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_ions.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_ions.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_ions.txt"
                end if
                    f_inout(i, k, l, m, n) = mms_solution_f_ions
            end if
        end do
        !$omp end do nowait
        end do
        end do
        end do
        end do
        !$omp end parallel
    end subroutine

end submodule
