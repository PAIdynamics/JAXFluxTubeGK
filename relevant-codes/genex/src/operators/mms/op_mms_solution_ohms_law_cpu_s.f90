submodule(op_mms_solution_m) op_mms_solution_ohms_law_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_nrad, &
                            get_minor_r, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_mms_ohm_cpu(this, mesh)
        class(op_mms_solution_ohms_law_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%mesh => mesh
    end subroutine

    module subroutine apply_mms_ohm_cpu(this, t, da_E_par_inout)
        class(op_mms_solution_ohms_law_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_2d_t), intent(inout) :: da_E_par_inout

        integer :: i, k, k_periodic, size_phi
        integer, dimension(2) :: lb, ub
        real(kind=GP) :: R, phi, Z
        real(kind=GP), contiguous, pointer, dimension(:,:) :: E_par_inout
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R_ptr, Z_ptr
        real(kind=GP), contiguous, pointer, dimension(:) :: phi_ptr

        integer :: npol, ntor, nrad
        ! MMS mode numbers
        real(kind=GP) :: omega, rhomin, rhomax, minorr, ellax1, ellax2
        ! MMS parameters
        real(kind=GP) :: mms_solution_e_par
        ! MMS solutions

        lb = da_E_par_inout%get_lbound()
        ub = da_E_par_inout%get_ubound()

        E_par_inout => da_E_par_inout%get_pointer()

        this%n_iterations = da_E_par_inout%get_size()
        call this%perf_counter%start_measurement()

        R_ptr   => this%mesh%get_R_pointer()
        Z_ptr   => this%mesh%get_Z_pointer()
        phi_ptr => this%mesh%get_phi_pointer()
        size_phi = this%mesh%size_phi()

        ! Set MMS parameters
        omega  = get_omega()
        npol   = get_npol()
        ntor   = get_ntor()
        nrad   = get_nrad()
        minorr = get_minor_r()
        ellax1 = get_ell_ax1()
        ellax2 = get_ell_ax2()
        rhomin = get_rho_min()
        rhomax = get_rho_max()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, t, npol, ntor, nrad, omega, &
        !$omp              rhomin, rhomax, minorr, size_phi, ellax1, ellax2) &
        !$omp shared(this, e_par_inout, R_ptr, Z_ptr, phi_ptr) &
        !$omp private(k, i, R, Z, phi, k_periodic, mms_solution_e_par)
        do k = lb(2), ub(2)
            ! NOTE: For correctness the MMS solution needs to be set at the
            !       ghost points as well, but the R, Z buffers are only given
            !       for the inner points in phi. Thus we need to use k_periodic.
            k_periodic = modulo(k - 1, size_phi) + 1
        !$omp do schedule(static)
        do i = lb(1), ub(1)

            R   = R_ptr(i, k_periodic)
            phi = phi_ptr(k_periodic)
            Z   = Z_ptr(i, k_periodic)

            if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_e_par.txt"
            else
include "../../mms/salpha/mms_solution_e_par.txt"
            endif
            e_par_inout(i, k) = mms_solution_e_par
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        call this%perf_counter%end_measurement()
    end subroutine

end submodule
