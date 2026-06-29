submodule (op_mms_source_m) op_mms_source_vlasov_eq_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_q, get_nrad, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_species_m, only: get_is_electrons
    use params_mesh_m, only: get_use_vspectral
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_vlasov_cpu(this, mesh, bsg_op)
        class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(in) :: mesh
        class(bsg_operators_t), target, intent(in) :: bsg_op

        this%mesh => mesh
        this%R    => this%mesh%get_R_pointer()
        this%Z    => this%mesh%get_Z_pointer()
        this%phi  => this%mesh%get_phi_pointer()
        this%vp   => this%mesh%get_vp_pointer()
        this%mu   => this%mesh%get_mu_pointer()
        this%bsg_op => bsg_op
        this%vp_bsg => this%bsg_op%get_vp_pointer()
        this%bsg_flags => this%mesh%get_bsg_flags_pointer()
    end subroutine

    module subroutine apply_vlasov_cpu(this, t, lb, ub, &
                                       lb_stripped, ub_stripped, f_out)
        class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        real(kind=GP), dimension(lb(1):ub(1), lb(2):ub(2), &
                                 lb(3):ub(3), lb(4):ub(4), &
                                 lb(5):ub(5)), intent(inout) :: f_out

        this%n_iterations = size(f_out, kind=INT64)
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call mms_source_vspec(this, t, lb, ub, &
                                  lb_stripped, ub_stripped, f_out)
        else
            call mms_source(this, t, lb, ub, &
                            lb_stripped, ub_stripped, f_out)
        endif

        call this%perf_counter%end_measurement()
    end subroutine

    subroutine mms_source(this, t, lb, ub, &
                          lb_stripped, ub_stripped, f_out)
        class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        real(kind=GP), dimension(lb(1):ub(1), lb(2):ub(2), &
                                 lb(3):ub(3), lb(4):ub(4), &
                                 lb(5):ub(5)), intent(inout) :: f_out

        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer :: i, k, l, m, n, ntor, npol, nrad
        real(kind=GP) :: omega, R, phi, Z, vp, mu, Lref, rhoref, q, &
                         mms_source_ions, mms_source_electrons, rhomin, &
                         rhomax, minorr, shear, dfdt, vpadv, bstaradv, &
                         bcrossadv, Bps, absB, dabsBdR, dabsBdphi, dabsBdZ, &
                         bR, bphi, bZ, dbRdphi, dbRdZ, dbphidR, dbphidZ, &
                         dbZdR, dbZdphi, ellax1, ellax2
        real(kind=GP) :: vpadv2, bcrossadv2
        logical :: is_electrons
        integer :: nb_point

        is_compute => this%mesh%get_is_compute_pointer()

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
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, f_out, is_compute) &
        !$omp private(n, m, l, k, i, R, Z, phi, &
        !$omp         vp, mu, mms_source_ions, is_electrons, &
        !$omp         mms_source_electrons, dfdt, vpadv, vpadv2, &
        !$omp         bstaradv, bcrossadv, bcrossadv2, &
        !$omp         Bps, absB, dabsBdR, dabsBdphi, dabsBdZ, &
        !$omp         bR, bphi, bZ, &
        !$omp         dbRdphi, dbRdZ, dbphidR, dbphidZ, dbZdR, dbZdphi, &
        !$omp         nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
            if(is_electrons) then
                ! Electrons
                if(this%mesh%equi_type() == SLAB) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/slab/mms_magfield.txt"

include "../../mms/slab/mms_source_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else if(this%mesh%equi_type() == CIRCULAR) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/circular/mms_magfield.txt"

include "../../mms/circular/mms_source_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else if(this%mesh%equi_type() == DOMMASCHK) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/dommaschk/mms_magfield.txt"

include "../../mms/dommaschk/mms_source_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/salpha/mms_magfield.txt"

include "../../mms/salpha/mms_source_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                endif
            else
                ! Ions
                if(this%mesh%equi_type() == SLAB) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/slab/mms_magfield.txt"

include "../../mms/slab/mms_source_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else if(this%mesh%equi_type() == CIRCULAR) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/circular/mms_magfield.txt"

include "../../mms/circular/mms_source_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else if(this%mesh%equi_type() == DOMMASCHK) then
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/dommaschk/mms_magfield.txt"

include "../../mms/dommaschk/mms_source_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                else
                    !$omp do simd schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        nb_point = unpack_bflag(this%bsg_flags(i))
                        vp  = this%vp_bsg(nb_point)%array(l)
                        mu  = this%mu(m)
include "../../mms/salpha/mms_magfield.txt"

include "../../mms/salpha/mms_source_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do simd nowait
                endif
            endif
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    subroutine mms_source_vspec(this, t, lb, ub, &
                                lb_stripped, ub_stripped, f_out)
        class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        integer, intent(in) :: lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        real(kind=GP), dimension(lb(1):ub(1), lb(2):ub(2), &
                                 lb(3):ub(3), lb(4):ub(4), &
                                 lb(5):ub(5)), intent(inout) :: f_out

        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer :: i, k, l, m, n, ntor, npol, nrad
        real(kind=GP) :: omega, R, phi, Z, vp, mu, Lref, rhoref, q, &
                         mms_source_ions, mms_source_electrons, rhomin, &
                         rhomax, minorr, shear, dfdt, vpadv, bstaradv, &
                         bcrossadv, Bps, absB, dabsBdR, dabsBdphi, dabsBdZ, &
                         bR, bphi, bZ, dbRdphi, dbRdZ, dbphidR, dbphidZ, &
                         dbZdR, dbZdphi
        real(kind=GP) :: a1, a2, a3, &
                         a4, a5, alkpj, &
                         b1, b2, b3, &
                         b4, blkpj
        real(kind=GP) :: dynamicvspec, tmaxw
        logical :: is_electrons

        is_compute => this%mesh%get_is_compute_pointer()

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

        ! NOTE: Vectorization with simd does not work currently for vspec, as
        !       the Intel ifx compiler generates errorneous code leading the
        !       mms test to fail. Removing simd fixes the problem.

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, tmaxw) &
        !$omp shared(this, f_out, is_compute) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, &
        !$omp         mms_source_ions, mms_source_electrons, &
        !$omp         absB, dabsBdR, dabsBdphi, dabsBdZ, bR, bphi, bZ, &
        !$omp         dbRdphi, dbRdZ, dbphidR, dbphidZ, &
        !$omp         dbZdR, dbZdphi, &
        !$omp         a1, a2, a3, a4, a5, alkpj, &
        !$omp         b1, b2, b3, b4, blkpj, &
        !$omp         dfdt, dynamicvspec, is_electrons)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
            if(is_electrons) then
                ! Electrons
                if(this%mesh%equi_type() == SLAB) then
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/slab/mms_magfield.txt"

include "../../mms/slab/mms_source_vspec_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                else if(this%mesh%equi_type() == CIRCULAR) then
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/circular/mms_magfield.txt"

include "../../mms/circular/mms_source_vspec_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                else
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/salpha/mms_magfield.txt"

include "../../mms/salpha/mms_source_vspec_electrons.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_electrons &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                endif
            else
                ! Ions
                if(this%mesh%equi_type() == SLAB) then
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/slab/mms_magfield.txt"

include "../../mms/slab/mms_source_vspec_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                else if(this%mesh%equi_type() == CIRCULAR) then
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/circular/mms_magfield.txt"

include "../../mms/circular/mms_source_vspec_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                else
                    !$omp do schedule(static)
                    do i = lb_stripped(1), ub_stripped(1)
                        R   = this%R(i, k)
                        phi = this%phi(k)
                        Z   = this%Z(i, k)
                        vp  = this%vp(l)
                        mu  = this%mu(m)
include "../../mms/salpha/mms_magfield.txt"

include "../../mms/salpha/mms_source_vspec_ions.txt"
                        f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                             + mms_source_ions &
                                             * is_compute(i, k)
                    enddo
                    !$omp end do nowait
                endif
            endif
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    ! NOTE: Fix for very poor performance with Intel ifx compiler. The
    !       interprocedural optimization seems to work worse than on ifort,
    !       such that this operator is very slow. The quick fix is adding these
    !       functions in this module.
    ! TODO: Remove and use the ones in math_m once the optimization is fixed.
    pure function krond(p, j) result(res)
        integer, intent(in) :: p
        integer, intent(in) :: j
        real(kind=GP) :: res
        res = 0.0_GP
        if(p == j) &
            res = 1.0_GP
    end function

    pure function sqrthv(a) result(res)
        real(kind=GP), intent(in) :: a
        real(kind=GP) :: res
        res = 0.0_GP
        if(a > 0.0_GP) &
            res = sqrt(a)
    end function

end submodule
