submodule (op_set_initial_condition_m) op_set_initial_condition_mms_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI, krond
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_species_m, only: get_is_electrons
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_q, get_nrad, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_mesh_m, only: get_use_vspectral
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_mms_cpu(this, mesh)
        class(op_set_initial_condition_mms_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(in) :: mesh

        this%mesh => mesh
    end subroutine

    module subroutine apply_mms_cpu(this, da_f_inout)
        class(op_set_initial_condition_mms_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        this%n_iterations = da_f_inout%get_size()
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call mms_set_init_vspec(this, da_f_inout)
        else
            call mms_set_init(this, da_f_inout)
        endif

        call this%perf_counter%end_measurement()

    end subroutine

    subroutine mms_set_init(this, da_f_inout)
        class(op_set_initial_condition_mms_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n, ierr
        integer, dimension(5) :: lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        integer :: npol, ntor, nrad
        real(kind=GP) :: omega, Lref, rhoref, q, rhomin, rhomax, minorr, &
                         shear, ellax1, ellax2
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons
        real(kind=GP), parameter :: t = 0.0_GP
        real(kind=GP) :: R, Z, phi, vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R_ptr, Z_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: vp_ptr, mu_ptr, &
                                                            phi_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: vp_bsg_ptr
        integer, contiguous, pointer, dimension(:) :: bsg_flags_pointer
        logical :: is_electrons
        integer :: nb_point

        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()

        mu_ptr       => this%mesh%get_mu_pointer()
        phi_ptr      => this%mesh%get_phi_pointer()
        R_ptr        => this%mesh%get_R_pointer()
        Z_ptr        => this%mesh%get_Z_pointer()
        bsg_flags_pointer => this%mesh%get_bsg_flags_pointer()

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
        !$omp firstprivate(lb_stripped, ub_stripped, q, &
        !$omp              npol, ntor, nrad, &
        !$omp              omega, Lref, rhoref, &
        !$omp              rhomin, rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, f_inout, R_ptr, Z_ptr, &
        !$omp        phi_ptr, mu_ptr, bsg_flags_pointer) &
        !$omp private(n, m, l, k, i, R, Z, &
        !$omp         phi, vp, mu, mms_solution_f_ions, &
        !$omp         mms_solution_f_electrons, is_electrons, nb_point, &
        !$omp         vp_bsg_ptr)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            R   = R_ptr(i, k)
            Z   = Z_ptr(i, k)
            phi = phi_ptr(k)
            nb_point = unpack_bflag(bsg_flags_pointer(i))
            vp_bsg_ptr => this%mesh%get_vp_pointer(nb_point)
            vp  = vp_bsg_ptr(l)
            mu  = mu_ptr(m)

            if(is_electrons) then
                ! Electrons
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
                ! Ions
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

    subroutine mms_set_init_vspec(this, da_f_inout)
        class(op_set_initial_condition_mms_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n, ierr, npol, ntor, nrad
        integer, dimension(5) :: lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP) :: omega, Lref, rhoref, q, rhomin, rhomax, minorr, shear
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons
        real(kind=GP), parameter :: t = 0.0_GP
        real(kind=GP) :: R, Z, phi, vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R_ptr, Z_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: vp_ptr, mu_ptr, &
                                                            phi_ptr
        logical :: is_electrons

        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()

        vp_ptr       => this%mesh%get_vp_pointer()
        mu_ptr       => this%mesh%get_mu_pointer()
        phi_ptr      => this%mesh%get_phi_pointer()
        R_ptr        => this%mesh%get_R_pointer()
        Z_ptr        => this%mesh%get_Z_pointer()

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
        !$omp firstprivate(lb_stripped, ub_stripped, &
        !$omp              q, npol, ntor, nrad, &
        !$omp              omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear) &
        !$omp shared(this, f_inout, R_ptr, &
        !$omp        Z_ptr, phi_ptr, vp_ptr, mu_ptr) &
        !$omp private(n, m, l, k, i, R, Z, &
        !$omp         phi, vp, mu, mms_solution_f_ions, &
        !$omp         mms_solution_f_electrons, is_electrons)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            R   = R_ptr(i, k)
            Z   = Z_ptr(i, k)
            phi = phi_ptr(k)
            vp  = vp_ptr(l)
            mu  = mu_ptr(m)
            if(is_electrons) then
                ! Electrons
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_electrons.txt"
                endif
                f_inout(i, k, l, m, n) = mms_solution_f_electrons
            else
                ! Ions
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
