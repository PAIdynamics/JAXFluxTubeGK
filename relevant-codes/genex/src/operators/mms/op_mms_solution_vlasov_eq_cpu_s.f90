submodule(op_mms_solution_m) op_mms_solution_vlasov_eq_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI, krond
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_q, get_nrad, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_species_m, only: get_is_electrons
    use params_mesh_m, only: get_use_vspectral, get_use_bsg
    use params_bsg_m, only: get_num_bsg_blocks
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    use array_generation_m, only: linspace
    implicit none

contains

    module subroutine initialize_mms_vlasov_cpu(this, dcomm_handler, mesh)
        class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        real(kind=GP) :: length
        real(kind=GP), contiguous, pointer, dimension(:) :: vp_ptr
        integer :: n_total_vp, n_ghost_vp, size_vp
        integer :: nb_point, num_blocks

        this%mesh => mesh
        size_vp = this%mesh%size_vp()
        n_ghost_vp = dcomm_handler%get_number_of_ghosts(3)
        n_total_vp = size_vp + 2.0_GP * n_ghost_vp

        ! Creates vp array that also includes ghost points. Assumes symmetric
        ! vp grid with entries going from negative to positive.
        ! Method generalized to BSG
        num_blocks = get_num_bsg_blocks()
        if (.not. get_use_bsg()) num_blocks = 1

        allocate(this%vp_total(num_blocks, &
                               (1 - n_ghost_vp):(size_vp + n_ghost_vp)))
        do nb_point = 1, num_blocks
            vp_ptr => this%mesh%get_vp_pointer(nb_point)
            length = -vp_ptr(1) &
                   + n_ghost_vp * this%mesh%delta_vp(nb_point)
            this%vp_total(nb_point, :) = &
                          linspace(-length, length, n_total_vp, endpoint=.true.)
        end do
    end subroutine

    module subroutine apply_mms_vlasov_cpu(this, t, da_f_inout)
        class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(inout) :: da_f_inout

        this%n_iterations = da_f_inout%get_size()
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call mms_solution_vspec(this, t, da_f_inout)
        else
            call mms_solution(this, t, da_f_inout)
        endif

        call this%perf_counter%end_measurement()

    end subroutine

    subroutine mms_solution(this, t, da_f_inout)
        class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(inout) :: da_f_inout
        logical :: is_electrons
        integer :: i, k, l, m, n, k_periodic, size_phi
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP) :: R, phi, Z, vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R_ptr, Z_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: phi_ptr, mu_ptr, &
                                                            vp_ptr
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout

        integer :: npol, ntor, nrad
        ! MMS mode numbers
        real(kind=GP) :: omega, Lref, rhoref, q, rhomin, rhomax, minorr, &
                         shear, ellax1, ellax2
        ! MMS parameters
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons
        ! MMS solutions
        integer :: nb_point
        integer, contiguous, pointer, dimension(:) :: bsg_flags

        bsg_flags => this%mesh%get_bsg_flags_pointer()

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()

        R_ptr   => this%mesh%get_R_pointer()
        Z_ptr   => this%mesh%get_Z_pointer()
        phi_ptr => this%mesh%get_phi_pointer()
        mu_ptr  => this%mesh%get_mu_pointer()
        vp_ptr  => this%mesh%get_vp_pointer()
        size_phi = this%mesh%size_phi()

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

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, lb, ub, size_phi, &
        !$omp              ellax1, ellax2) &
        !$omp shared(f_inout, this, R_ptr, phi_ptr, Z_ptr, mu_ptr, &
        !$omp        bsg_flags) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, k_periodic, &
        !$omp         is_electrons, mms_solution_f_electrons, &
        !$omp         mms_solution_f_ions, nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb(3), ub(3)
        do k = lb(2), ub(2)
            ! NOTE: For correctness the MMS solution needs
            !       to be set at the ghost points as well,
            !       but the R, Z buffers are only given
            !       for the inner points in phi. Thus we
            !       need to use k_periodic.
            k_periodic = modulo(k - 1, size_phi) + 1
        !$omp do schedule(static)
        do i = lb(1), ub(1)

            R   = R_ptr(i, k_periodic)
            phi = phi_ptr(k_periodic)
            Z   = Z_ptr(i, k_periodic)
            nb_point = unpack_bflag(bsg_flags(i))
            vp = this%vp_total(nb_point, l)
            mu  = mu_ptr(m)

            if(is_electrons) then
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_electrons.txt"
                else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_f_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_electrons.txt"
                endif
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
                endif
                f_inout(i, k, l, m, n) = mms_solution_f_ions
            endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    subroutine mms_solution_vspec(this, t, da_f_inout)
        class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(inout) :: da_f_inout
        logical :: is_electrons
        integer :: i, k, l, m, n, k_periodic, size_phi
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP) :: R, phi, Z, vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R_ptr, Z_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: phi_ptr, mu_ptr, &
                                                            vp_ptr
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout

        integer :: npol, ntor, nrad
        ! MMS mode numbers
        real(kind=GP) :: omega, Lref, rhoref, q, rhomin, rhomax, minorr, shear
        ! MMS parameters
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons
        ! MMS solutions

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()

        R_ptr   => this%mesh%get_R_pointer()
        Z_ptr   => this%mesh%get_Z_pointer()
        phi_ptr => this%mesh%get_phi_pointer()
        mu_ptr  => this%mesh%get_mu_pointer()
        vp_ptr  => this%mesh%get_vp_pointer()
        size_phi = this%mesh%size_phi()

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

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, lb, ub, size_phi) &
        !$omp shared(f_inout, this, R_ptr, phi_ptr, &
        !$omp        Z_ptr, vp_ptr, mu_ptr) &
        !$omp private(n, m, l, k, k_periodic, i, R, Z, phi, vp, mu, &
        !$omp         mms_solution_f_electrons, mms_solution_f_ions, &
        !$omp         is_electrons)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb(2), ub(2)
            ! NOTE: For correctness the MMS solution needs
            !       to be set at the ghost points in RZ and phi
            !       only for the spectral approach.
            k_periodic = modulo(k - 1, size_phi) + 1
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            R   = R_ptr(i, k_periodic)
            phi = phi_ptr(k_periodic)
            Z   = Z_ptr(i, k_periodic)
            vp  = vp_ptr(l)
            mu  = mu_ptr(m)

            if(is_electrons) then
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_electrons.txt"
                endif
                f_inout(i, k, l, m, n) = mms_solution_f_electrons
            else
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_ions.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_ions.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_ions.txt"
                endif
                f_inout(i, k, l, m, n) = mms_solution_f_ions
            endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine
end submodule
