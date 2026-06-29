submodule (op_mms_source_m) op_mms_source_maxwells_eq_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_beta_ref
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_q, get_nrad, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_mesh_m, only: get_use_vspectral
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_em_fields_cpu(this, mesh, bsg_op)
        class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(in) :: mesh
        class(bsg_operators_t), target, intent(in) :: bsg_op

        this%mesh => mesh
        this%R    => this%mesh%get_R_pointer()
        this%Z    => this%mesh%get_Z_pointer()
        this%phi  => this%mesh%get_phi_pointer()
        this%vp   => this%mesh%get_vp_pointer()
        this%mu   => this%mesh%get_mu_pointer()
    end subroutine

    module subroutine apply_em_fields_cpu(this, t, lb, ub, lb_stripped, &
                                          ub_stripped, b_qn_eq, b_amps_law, &
                                          b_bpar_eq)
        class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(2), ub(2), lb_stripped(2), ub_stripped(2)
        real(kind=GP), intent(inout) :: &
            b_qn_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_amps_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_bpar_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))

        this%n_iterations = size(b_qn_eq, kind=INT64)
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call mms_source_vspec(this, t, lb, ub, lb_stripped, &
                                  ub_stripped, b_qn_eq, b_amps_law, b_bpar_eq)
        else
            call mms_source(this, t, lb, ub, lb_stripped, &
                            ub_stripped, b_qn_eq, b_amps_law, b_bpar_eq)
        endif

        call this%perf_counter%end_measurement()

    end subroutine

    subroutine mms_source(this, t, lb, ub, lb_stripped, &
                          ub_stripped, b_qn_eq, b_amps_law, b_bpar_eq)
        class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(2), ub(2), lb_stripped(2), ub_stripped(2)
        real(kind=GP), intent(inout) :: &
            b_qn_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_amps_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_bpar_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))

        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer :: i, k, ntor, npol, nrad
        real(kind=GP) :: omega, R, phi, Z, Lref, rhoref, q, betaref, &
                         mms_source_es_pot, rhomin, rhomax, minorr, shear, &
                         mms_source_a_par, mms_source_b_par, absB, dabsBdR, &
                         dabsBdphi, dabsBdZ, bR, bphi, bZ, dbRdphi, dbRdZ, &
                         dbphidR, dbphidZ, dbZdR, dbZdphi, ellax1, ellax2

        is_compute => this%mesh%get_is_compute_pointer()

        ! Set MMS parameters
        betaref = get_beta_ref()
        rhoref  = get_rho_ref()
        Lref    = get_L_ref()
        omega   = get_omega()
        q       = get_q()
        npol    = get_npol()
        ntor    = get_ntor()
        nrad    = get_nrad()
        minorr  = get_minor_r()
        shear   = get_shear()
        ellax1  = get_ell_ax1()
        ellax2  = get_ell_ax2()
        rhomin  = get_rho_min()
        rhomax  = get_rho_max()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, t, q, &
        !$omp              npol, ntor, nrad, omega, Lref, rhoref, &
        !$omp              rhomin, rhomax, minorr, shear, betaref, &
        !$omp              ellax1, ellax2) &
        !$omp shared(this, b_qn_eq, b_amps_law, b_bpar_eq, is_compute) &
        !$omp private(k, i, R, Z, phi, &
        !$omp         mms_source_es_pot, mms_source_a_par, mms_source_b_par, &
        !$omp         absB, dabsBdR, dabsBdphi, dabsBdZ, &
        !$omp         bR, bphi, bZ, dbRdphi, dbRdZ, &
        !$omp         dbphidR, dbphidZ, dbZdR, dbZdphi)
        do k = lb_stripped(2), ub_stripped(2)
            if(this%mesh%equi_type() == SLAB) then
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/slab/mms_magfield.txt"

include "../../mms/slab/mms_source_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)
include "../../mms/slab/mms_source_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
include "../../mms/slab/mms_source_b_par.txt"
                    b_bpar_eq(i, k) = b_bpar_eq(i, k) &
                                     + mms_source_b_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            else if(this%mesh%equi_type() == CIRCULAR) then
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/circular/mms_magfield.txt"

include "../../mms/circular/mms_source_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)
include "../../mms/circular/mms_source_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
include "../../mms/circular/mms_source_b_par.txt"
                    b_bpar_eq(i, k) = b_bpar_eq(i, k) &
                                     + mms_source_b_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            else if(this%mesh%equi_type() == DOMMASCHK) then
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/dommaschk/mms_magfield.txt"

include "../../mms/dommaschk/mms_source_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)
include "../../mms/dommaschk/mms_source_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
include "../../mms/dommaschk/mms_source_b_par.txt"
                    b_bpar_eq(i, k) = b_bpar_eq(i, k) &
                                     + mms_source_b_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            else
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/salpha/mms_magfield.txt"

include "../../mms/salpha/mms_source_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)
include "../../mms/salpha/mms_source_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
include "../../mms/salpha/mms_source_b_par.txt"
                    b_bpar_eq(i, k) = b_bpar_eq(i, k) &
                                     + mms_source_b_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            endif
        enddo
        !$omp end parallel
    end subroutine

    subroutine mms_source_vspec(this, t, lb, ub, lb_stripped, &
                                ub_stripped, b_qn_eq, b_amps_law, b_bpar_eq)
        class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(2), ub(2), lb_stripped(2), ub_stripped(2)
        real(kind=GP), intent(inout) :: &
            b_qn_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_amps_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        real(kind=GP), intent(inout) :: &
            b_bpar_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        ! TODO: add B_par contributions

        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer :: i, k, ntor, npol, nrad
        real(kind=GP) :: omega, R, phi, Z, Lref, rhoref, q, betaref, &
                         mms_source_es_pot, rhomin, rhomax, minorr, shear, &
                         mms_source_a_par, absB, dabsBdR, dabsBdphi, dabsBdZ, &
                         bR, bphi, bZ, dbRdphi, dbRdZ, dbphidR, dbphidZ, &
                         dbZdR, dbZdphi

        is_compute => this%mesh%get_is_compute_pointer()

        ! Set MMS parameters
        betaref = get_beta_ref()
        rhoref  = get_rho_ref()
        Lref    = get_L_ref()
        omega   = get_omega()
        q       = get_q()
        npol    = get_npol()
        ntor    = get_ntor()
        nrad    = get_nrad()
        minorr  = get_minor_r()
        shear   = get_shear()
        rhomin  = this%mesh%rho_min()
        rhomax  = this%mesh%rho_max()

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, t, q, &
        !$omp              npol, ntor, nrad, omega, Lref, rhoref, &
        !$omp              rhomin, rhomax, minorr, shear, betaref) &
        !$omp shared(this, b_qn_eq, b_amps_law, is_compute) &
        !$omp private(k, i, R, Z, phi, mms_source_es_pot, &
        !$omp         mms_source_a_par, &
        !$omp         absB, dabsBdR, dabsBdphi, dabsBdZ, &
        !$omp         bR, bphi, bZ, &
        !$omp         dbRdphi, dbRdZ, dbphidR, dbphidZ, &
        !$omp         dbZdR, dbZdphi)
        do k = lb_stripped(2), ub_stripped(2)
            if(this%mesh%equi_type() == SLAB) then
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/slab/mms_magfield.txt"
include "../../mms/slab/mms_source_vspec_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)

include "../../mms/slab/mms_source_vspec_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            else if(this%mesh%equi_type() == CIRCULAR) then
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/circular/mms_magfield.txt"
include "../../mms/circular/mms_source_vspec_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)
include "../../mms/circular/mms_source_vspec_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            else
                !$omp do simd schedule(static)
                do i = lb_stripped(1), ub_stripped(1)
                    R   = this%R(i, k)
                    phi = this%phi(k)
                    Z   = this%Z(i, k)

include "../../mms/salpha/mms_magfield.txt"
include "../../mms/salpha/mms_source_vspec_es_pot.txt"
                    b_qn_eq(i, k) = b_qn_eq(i, k) &
                                  + mms_source_es_pot &
                                  * is_compute(i, k)

include "../../mms/salpha/mms_source_vspec_a_par.txt"
                    b_amps_law(i, k) = b_amps_law(i, k) &
                                     + mms_source_a_par &
                                     * is_compute(i, k)
                enddo
                !$omp end do simd nowait
            endif
        enddo
        !$omp end parallel
    end subroutine
end submodule
