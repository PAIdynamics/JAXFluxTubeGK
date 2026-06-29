submodule (op_mms_error_m) op_mms_error_cpu_s
    use mpi
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: MPI_GP, GP_EPS
    use math_m, only: PI, krond
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_mms_m, only: get_omega, get_npol, get_ntor, get_nrad, get_q, &
                            get_minor_r, get_shear, get_ell_ax1, get_ell_ax2, &
                            get_rho_min, get_rho_max
    use params_species_m, only: get_is_electrons
    use params_mesh_m, only: get_use_vspectral
    use bsg_types_m, only: unpack_bflag
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, DOMMASCHK
    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
        class(op_mms_error_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), target, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(in) :: mesh
        class(bsg_operators_t), target, intent(in) :: bsg_op

        this%mesh => mesh
        this%bsg_op => bsg_op

        this%dcomm_handler => dcomm_handler

        this%R   => this%mesh%get_R_pointer()
        this%Z   => this%mesh%get_Z_pointer()
        this%phi => this%mesh%get_phi_pointer()
        this%vp  => this%mesh%get_vp_pointer()
        this%mu  => this%mesh%get_mu_pointer()
        this%vpw => this%mesh%get_vpw_pointer()
        this%muw => this%mesh%get_muw_pointer()
        this%phiw => this%mesh%get_phiw_pointer()
        this%rzw => this%mesh%get_rzw_pointer()
        this%vp_bsg => this%bsg_op%get_vp_pointer()
        this%vpw_bsg => this%bsg_op%get_vpw_pointer()
        this%bsg_flags => this%mesh%get_bsg_flags_pointer()

    end subroutine

    module subroutine apply_cpu(this, t, da_f_in, da_phi_in, da_A_par_in, &
                                da_B_par_in, da_E_par_in, l2_err, linf_err)
        class(op_mms_error_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in, da_A_par_in, &
                                              da_B_par_in, da_E_par_in
        real(kind=GP), dimension(6), intent(out) :: l2_err
        real(kind=GP), dimension(6), intent(out) :: linf_err

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        if(get_use_vspectral()) then
            call mms_error_vspec(this, t, da_f_in, da_phi_in, da_A_par_in, &
                                 da_B_par_in, da_E_par_in, l2_err, linf_err)
        else
            call mms_error(this, t, da_f_in, da_phi_in, da_A_par_in, &
                           da_B_par_in, da_E_par_in, l2_err, linf_err)
        endif

        call this%perf_counter%end_measurement()
    end subroutine

    subroutine mms_error(this, t, da_f_in, da_phi_in, da_A_par_in, &
                         da_B_par_in, da_E_par_in, l2_err, linf_err)
        class(op_mms_error_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in, da_A_par_in, &
                                              da_B_par_in, da_E_par_in
        real(kind=GP), dimension(6), intent(out) :: l2_err
        real(kind=GP), dimension(6), intent(out) :: linf_err
        integer :: i, j, k, l, m, n
        ! Loop indices
        integer :: ntor, npol, nrad
        ! MMS mode numbers
        integer :: ierr
        ! Error code
        integer, dimension(5) :: lb_stripped, ub_stripped
        ! Lower and upper bound of the domain without the ghost
        real(kind=GP) :: omega, Lref, rhoref, q, weight, rhomin, rhomax, &
                         minorr, shear, ellax1, ellax2
        ! MMS parameters
        real(kind=GP) :: R, phi, Z, vp, mu
        ! Coordinates
        real(kind=GP), dimension(:,:), contiguous, pointer :: is_compute
        ! Pointer to compute mask
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons, &
                         mms_solution_es_pot, mms_solution_a_par, &
                         mms_solution_b_par, mms_solution_e_par
        ! MMS solutions
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: f_in
        real(kind=GP), pointer, dimension(:,:) :: phi_in, A_par_in, B_par_in, &
                                                  E_par_in
        real(kind=GP), dimension(4) :: l2_err_local, linf_err_local, &
                                       l2_norm_local, linf_norm_local
        ! Buffers to sum the L2 and Linf errors and norms on the local MPI
        ! processes. We need 4 dimensions to store values for es_pot, A_par,
        ! B_par and E_par in one go.
        real(kind=GP), dimension(6) :: l2_norm, linf_norm
        ! Buffers to calculate the L2 and Linf norms
        logical :: is_electrons
        integer :: nb_point

        is_compute => this%mesh%get_is_compute_pointer()

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer_stripped()
        phi_in => da_phi_in%get_readonly_pointer_stripped()
        A_par_in => da_A_par_in%get_readonly_pointer_stripped()
        B_par_in => da_B_par_in%get_readonly_pointer_stripped()
        E_par_in => da_E_par_in%get_readonly_pointer_stripped()

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

        ! Initialize MMS errors and norms
        l2_err    = 0.0_GP
        l2_norm   = 0.0_GP
        linf_err  = 0.0_GP
        linf_norm = 0.0_GP

        ! Calculate L2 and Linf error of the distribution function
        l2_err_local    = 0.0_GP
        l2_norm_local   = 0.0_GP
        linf_err_local  = 0.0_GP
        linf_norm_local = 0.0_GP

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, f_in, l2_err_local, &
        !$omp        l2_norm_local, linf_err_local, &
        !$omp        linf_norm_local, is_compute) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, nb_point, &
        !$omp         mms_solution_f_ions, &
        !$omp         mms_solution_f_electrons, weight, is_electrons)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static) &
        !$omp reduction(+:l2_err_local, l2_norm_local) &
        !$omp reduction(max:linf_err_local, linf_norm_local)
        do i = lb_stripped(1), ub_stripped(1)
        if(is_compute(i, k) == 1.0_GP) then
            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            nb_point = unpack_bflag(this%bsg_flags(i))
            vp = this%vp_bsg(nb_point)%array(l)
            mu  = this%mu(m)
            weight = this%vpw_bsg(nb_point)%array(l) * this%muw(m) &
                   * this%phiw(k) * this%rzw(i, k)
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
                l2_err_local(n) = l2_err_local(n) &
                                + weight &
                                * (mms_solution_f_electrons &
                                   - f_in(i, k, l, m, n))**2
                l2_norm_local(n) = l2_norm_local(n) &
                                     + weight &
                                     * mms_solution_f_electrons**2
                linf_err_local(n) = max(linf_err_local(n), &
                                        abs(mms_solution_f_electrons &
                                            - f_in(i, k, l, m, n)))
                linf_norm_local(n) = max(linf_norm_local(n), &
                                         mms_solution_f_electrons)
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

                l2_err_local(n) = l2_err_local(n) &
                                + weight &
                                * (mms_solution_f_ions &
                                   - f_in(i, k, l, m, n))**2
                l2_norm_local(n) = l2_norm_local(n) &
                                 + weight &
                                 * mms_solution_f_ions**2
                linf_err_local(n) = max(linf_err_local(n), &
                                        abs(mms_solution_f_ions &
                                            - f_in(i, k, l, m, n)))
                linf_norm_local(n) = max(linf_norm_local(n), &
                                        mms_solution_f_ions)
            endif
        endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        ! Communicate L2 and Linf error
        block
            ! We do the MPI reduction over 4 elements because the local
            ! buffer has this size. Because in this calculation
            ! we only consider the error of ions and electrons
            ! we will only use the first 2 elements afterwards.
            real(kind=GP), dimension(4) :: l2_err_recv, l2_norm_recv, &
                                           linf_err_recv, linf_norm_recv
            ! Receive buffers for L2 and Linf error and norm
             call MPI_Reduce(l2_err_local, l2_err_recv, 4, MPI_GP, &
                            MPI_SUM, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(l2_norm_local, l2_norm_recv, 4, MPI_GP, &
                            MPI_SUM, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(linf_err_local, linf_err_recv, 4, MPI_GP, &
                            MPI_MAX, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(linf_norm_local, linf_norm_recv, &
                            4, MPI_GP, &
                            MPI_MAX, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)

            if(this%dcomm_handler%is_master()) then
                do j = 1, 2
                    ! Multiply common RZ and phi weights and
                    ! take the square root for L2 norm
                    l2_err(j)  = sqrt(l2_err_recv(j))
                    l2_norm(j) = sqrt(l2_norm_recv(j))
                    l2_err(j)  = l2_err(j) / (l2_norm(j) + GP_EPS)
                    linf_err(j)  = linf_err_recv(j)
                    linf_norm(j) = linf_norm_recv(j)
                    linf_err(j)  = linf_err(j) &
                                 / (linf_norm(j) + GP_EPS)
                enddo
            endif
        end block

        ! Calculate L2 and Linf error of the electrostatic potential,
        ! the parallel electromagnetic vector potential, the parallel
        ! magnetic fluctuations and the parallel electric field

        ! We need to reset local error and norm buffers for next calculation
        l2_err_local    = 0.0_GP
        l2_norm_local   = 0.0_GP
        linf_err_local  = 0.0_GP
        linf_norm_local = 0.0_GP

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, is_compute, phi_in, A_par_in, B_par_in, E_par_in, &
        !$omp        l2_err_local, l2_norm_local, linf_err_local, &
        !$omp        linf_norm_local) &
        !$omp private(k, i, R, Z, phi, weight, mms_solution_es_pot, &
        !$omp         mms_solution_a_par, mms_solution_b_par, &
        !$omp         mms_solution_e_par)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static) &
        !$omp reduction(+:l2_err_local, l2_norm_local) &
        !$omp reduction(max:linf_err_local, linf_norm_local)
        do i = lb_stripped(1), ub_stripped(1)
        if(is_compute(i, k) == 1.0_GP) then
            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            weight = this%phiw(k) * this%rzw(i, k)

            if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_es_pot.txt"
include "../../mms/slab/mms_solution_a_par.txt"
include "../../mms/slab/mms_solution_b_par.txt"
include "../../mms/slab/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_es_pot.txt"
include "../../mms/circular/mms_solution_a_par.txt"
include "../../mms/circular/mms_solution_b_par.txt"
include "../../mms/circular/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_es_pot.txt"
include "../../mms/dommaschk/mms_solution_a_par.txt"
include "../../mms/dommaschk/mms_solution_b_par.txt"
include "../../mms/dommaschk/mms_solution_e_par.txt"
            else
include "../../mms/salpha/mms_solution_es_pot.txt"
include "../../mms/salpha/mms_solution_a_par.txt"
include "../../mms/salpha/mms_solution_b_par.txt"
include "../../mms/salpha/mms_solution_e_par.txt"
            endif

            ! Electrostatic potential
            l2_err_local(1)    = l2_err_local(1) &
                               + weight &
                                * (mms_solution_es_pot - phi_in(i, k))**2
            l2_norm_local(1)   = l2_norm_local(1) &
                               + weight &
                                * mms_solution_es_pot**2
            linf_err_local(1)  = max(linf_err_local(1), &
                                     abs(mms_solution_es_pot - phi_in(i, k)))
            linf_norm_local(1) = max(linf_norm_local(1), mms_solution_es_pot)

            ! Parallel electromagnetic vector potential
            l2_err_local(2)    = l2_err_local(2) &
                               + weight &
                                * (mms_solution_a_par - A_par_in(i, k))**2
            l2_norm_local(2)   = l2_norm_local(2) &
                               + weight &
                                * mms_solution_a_par**2
            linf_err_local(2)  = max(linf_err_local(2), &
                                     abs(mms_solution_a_par - A_par_in(i, k)))
            linf_norm_local(2) = max(linf_norm_local(2), mms_solution_a_par)

            ! Parallel magnetic fluctuations
            l2_err_local(3)    = l2_err_local(3) &
                               + weight &
                                * (mms_solution_b_par - B_par_in(i, k))**2
            l2_norm_local(3)   = l2_norm_local(3) &
                               + weight &
                                * mms_solution_b_par**2
            linf_err_local(3)  = max(linf_err_local(3), &
                                     abs(mms_solution_b_par - B_par_in(i, k)))
            linf_norm_local(3) = max(linf_norm_local(3), mms_solution_b_par)

            ! Parallel electric field
            l2_err_local(4)    = l2_err_local(4) &
                               + weight &
                                * (mms_solution_e_par - E_par_in(i, k))**2
            l2_norm_local(4)   = l2_norm_local(4) &
                               + weight &
                                * mms_solution_e_par**2
            linf_err_local(4)  = max(linf_err_local(4), &
                                     abs(mms_solution_e_par - E_par_in(i, k)))
            linf_norm_local(4) = max(linf_norm_local(4), mms_solution_e_par)
        endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        ! Communicate L2 and Linf error
        block
            real(kind=GP), dimension(4) :: l2_err_recv, l2_norm_recv, &
                                           linf_err_recv, linf_norm_recv
            ! Receive buffers for L2 and Linf error and norm

            call MPI_Reduce(l2_err_local, l2_err_recv, 4, MPI_GP, &
                            MPI_SUM, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(l2_norm_local, l2_norm_recv, 4, MPI_GP, &
                            MPI_SUM, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(linf_err_local, linf_err_recv, 4, MPI_GP, &
                            MPI_MAX, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(linf_norm_local, linf_norm_recv, 4, MPI_GP, &
                            MPI_MAX, 0, this%dcomm_handler%get_comm_phi(), ierr)

            if(this%dcomm_handler%is_master()) then
                do j = 3, 6
                    ! Multiply common RZ and phi weights and
                    ! take the square root for L2 norm
                    l2_err(j)  = sqrt(l2_err_recv(j - 2))
                    l2_norm(j) = sqrt(l2_norm_recv(j - 2))
                    l2_err(j)  = l2_err(j) / (l2_norm(j) + GP_EPS)

                    linf_err(j)  = linf_err_recv(j - 2)
                    linf_norm(j) = linf_norm_recv(j - 2)
                    linf_err(j)  = linf_err(j) / (linf_norm(j) + GP_EPS)
                enddo
            endif
        end block
    end subroutine

    subroutine mms_error_vspec(this, t, da_f_in, da_phi_in, da_A_par_in, &
                            da_B_par_in, da_E_par_in, l2_err, linf_err)
        class(op_mms_error_cpu_t), intent(inout) :: this
        real(kind=GP), intent(in) :: t
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in, da_A_par_in, &
                                              da_B_par_in, da_E_par_in
        real(kind=GP), dimension(6), intent(out) :: l2_err
        real(kind=GP), dimension(6), intent(out) :: linf_err

        integer :: i, j, k, l, m, n
        ! Loop indices
        integer :: ierr
        ! Error code
        integer, dimension(5) :: lb_stripped, ub_stripped
        ! Lower and upper bound of the domain without the ghost
        integer :: ntor, npol, nrad
        ! MMS mode numbers
        real(kind=GP) :: omega, Lref, rhoref, q, weight, rhomin, rhomax, &
                         minorr, shear, ellax1, ellax2
        ! MMS parameters
        real(kind=GP) :: R, phi, Z, vp, mu
        ! Coordinates
        real(kind=GP), dimension(:,:), contiguous, pointer :: is_compute
        ! Pointer to compute mask
        real(kind=GP) :: mms_solution_f_ions, mms_solution_f_electrons, &
                         mms_solution_es_pot, mms_solution_a_par, &
                         mms_solution_e_par
        ! MMS solutions
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: f_in
        real(kind=GP), pointer, dimension(:,:) :: phi_in, A_par_in, E_par_in
        real(kind=GP), dimension(4) :: l2_err_local, linf_err_local, &
                                       l2_norm_local, linf_norm_local
        ! Buffers to sum the L2 and Linf errors and norms on the local MPI
        ! processes. We need 4 dimensions to store values for es_pot, A_par,
        ! B_par and E_par in one go.
        real(kind=GP), dimension(6) :: l2_norm, linf_norm
        ! Buffers to calculate the L2 and Linf norms
        real(kind=GP) :: weight_l2, weight_inf
        ! Weights of the L2 and inf norms
        logical :: is_electrons

        is_compute => this%mesh%get_is_compute_pointer()

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer_stripped()
        phi_in => da_phi_in%get_readonly_pointer_stripped()
        A_par_in => da_A_par_in%get_readonly_pointer_stripped()
        E_par_in => da_E_par_in%get_readonly_pointer_stripped()

        ! Set MMS parameters
        rhoref = get_rho_ref()
        Lref   = get_L_ref()
        omega  = get_omega()
        q      = get_q()
        npol   = get_npol()
        ntor   = get_ntor()
        nrad   = get_nrad()
        ellax1 = get_ell_ax1()
        ellax2 = get_ell_ax2()
        minorr = get_minor_r()
        shear  = get_shear()
        rhomin = this%mesh%rho_min()
        rhomax = this%mesh%rho_max()

        ! Initialize MMS errors and norms
        l2_err    = 0.0_GP
        l2_norm   = 0.0_GP
        linf_err  = 0.0_GP
        linf_norm = 0.0_GP

        ! Calculate L2 and Linf error of the distribution function
        l2_err_local    = 0.0_GP
        l2_norm_local   = 0.0_GP
        linf_err_local  = 0.0_GP
        linf_norm_local = 0.0_GP

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, f_in, l2_err_local, &
        !$omp        l2_norm_local, linf_err_local, &
        !$omp        linf_norm_local, is_compute) &
        !$omp private(n, m, l, k, i, R, Z, phi, vp, mu, &
        !$omp         mms_solution_f_ions, &
        !$omp         mms_solution_f_electrons, &
        !$omp         weight_l2, weight_inf, is_electrons)
        do n = lb_stripped(5), ub_stripped(5)
        is_electrons = get_is_electrons(n)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static) &
        !$omp reduction(+:l2_err_local, l2_norm_local) &
        !$omp reduction(max:linf_err_local, linf_norm_local)
        do i = lb_stripped(1), ub_stripped(1)
        if(is_compute(i, k) == 1.0_GP) then

            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            vp  = this%vp(l)
            mu  = this%mu(m)

            ! NOTE: the inf error is evaluated for the density only
            weight_inf = this%vpw(l) * this%muw(m)

            weight_l2 = this%vpw(l) * this%muw(m) &
                      * this%phiw(k) * this%rzw(i, k)

            if(is_electrons) then
                ! Electrons
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_electrons.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_electrons.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_electrons.txt"
                endif

                l2_err_local(n) = l2_err_local(n) &
                                + weight_l2 &
                                    * (mms_solution_f_electrons &
                                    - f_in(i, k, l, m, n))**2
                l2_norm_local(n) = l2_norm_local(n) &
                                 + weight_l2 &
                                 * mms_solution_f_electrons**2
                linf_err_local(n) = max(linf_err_local(n), &
                                        weight_inf &
                                        * abs(mms_solution_f_electrons &
                                        - f_in(i, k, l, m, n)))
                linf_norm_local(n) = max(linf_norm_local(n), &
                                        weight_inf &
                                        * mms_solution_f_electrons)
            else
                ! Ions
                if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_f_vspec_ions.txt"
                else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_f_vspec_ions.txt"
                else
include "../../mms/salpha/mms_solution_f_vspec_ions.txt"
                endif

                l2_err_local(n)    = l2_err_local(n) &
                                + weight_l2 &
                                * (mms_solution_f_ions &
                                - f_in(i, k, l, m, n))**2
                l2_norm_local(n)   = l2_norm_local(n) &
                                + weight_l2 &
                                * mms_solution_f_ions**2
                linf_err_local(n)  = max(linf_err_local(n), &
                                        weight_inf * &
                                        abs(mms_solution_f_ions &
                                        - f_in(i, k, l, m, n)))
                linf_norm_local(n) = max(linf_norm_local(n), &
                                        weight_inf &
                                        * mms_solution_f_ions)
            endif
        endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        ! Communicate L2 and Linf error
        block
            ! We do the MPI reduction over 4 elements because the local
            ! buffer has this size. Because in this calculation
            ! we only consider the error of ions and electrons
            ! we will only use the first 2 elements afterwards.
            real(kind=GP), dimension(4) :: l2_err_recv, l2_norm_recv, &
                                           linf_err_recv, linf_norm_recv
            ! Receive buffers for L2 and Linf error and norm

            call MPI_Reduce(l2_err_local, l2_err_recv, 4, MPI_GP, &
                            MPI_SUM, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(l2_norm_local, l2_norm_recv, 4, MPI_GP, &
                            MPI_SUM, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(linf_err_local, linf_err_recv, 4, &
                            MPI_GP, MPI_MAX, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)
            call MPI_Reduce(linf_norm_local, linf_norm_recv, 4, &
                            MPI_GP, &
                            MPI_MAX, 0, &
                            this%dcomm_handler%get_comm_cart(), &
                            ierr)

            if(this%dcomm_handler%is_master()) then
                do j = 1, 2
                    ! Multiply common RZ and phi weights and
                    ! take the square root for L2 norm
                    l2_err(j)  = sqrt(l2_err_recv(j))
                    l2_norm(j) = sqrt(l2_norm_recv(j))
                    l2_err(j)  = l2_err(j) / (l2_norm(j) + GP_EPS)

                    linf_err(j)  = linf_err_recv(j)
                    linf_norm(j) = linf_norm_recv(j)
                    linf_err(j)  = linf_err(j) / (linf_norm(j) + GP_EPS)
                enddo
            endif
        end block

        ! Calculate L2 and Linf error of the electrostatic potential,
        ! the parallel electromagnetic vector potential and the parallel
        ! electric field

        ! We need to reset local error and norm buffers for next calculation
        l2_err_local    = 0.0_GP
        l2_norm_local   = 0.0_GP
        linf_err_local  = 0.0_GP
        linf_norm_local = 0.0_GP

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, t, q, npol, &
        !$omp              ntor, nrad, omega, Lref, rhoref, rhomin, &
        !$omp              rhomax, minorr, shear, ellax1, ellax2) &
        !$omp shared(this, is_compute, phi_in, A_par_in, E_par_in, &
        !$omp        l2_err_local, l2_norm_local, linf_err_local, &
        !$omp        linf_norm_local) &
        !$omp private(k, i, R, Z, phi, weight, mms_solution_es_pot, &
        !$omp         mms_solution_a_par, mms_solution_e_par)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static) &
        !$omp reduction(+:l2_err_local, l2_norm_local) &
        !$omp reduction(max:linf_err_local, linf_norm_local)
        do i = lb_stripped(1), ub_stripped(1)
        if(is_compute(i, k) == 1.0_GP) then
            R   = this%R(i, k)
            phi = this%phi(k)
            Z   = this%Z(i, k)
            weight = this%phiw(k) * this%rzw(i, k)

            if(this%mesh%equi_type() == SLAB) then
include "../../mms/slab/mms_solution_es_pot.txt"
include "../../mms/slab/mms_solution_a_par.txt"
include "../../mms/slab/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == CIRCULAR) then
include "../../mms/circular/mms_solution_es_pot.txt"
include "../../mms/circular/mms_solution_a_par.txt"
include "../../mms/circular/mms_solution_e_par.txt"
            else if(this%mesh%equi_type() == DOMMASCHK) then
include "../../mms/dommaschk/mms_solution_es_pot.txt"
include "../../mms/dommaschk/mms_solution_a_par.txt"
include "../../mms/dommaschk/mms_solution_e_par.txt"
            else
include "../../mms/salpha/mms_solution_es_pot.txt"
include "../../mms/salpha/mms_solution_a_par.txt"
include "../../mms/salpha/mms_solution_e_par.txt"
            endif

            ! Electrostatic potential
            l2_err_local(1)    = l2_err_local(1) &
                               + weight &
                                * (mms_solution_es_pot - phi_in(i, k))**2
            l2_norm_local(1)   = l2_norm_local(1) &
                               + weight &
                                * mms_solution_es_pot**2
            linf_err_local(1)  = max(linf_err_local(1), &
                                     abs(mms_solution_es_pot - phi_in(i, k)))
            linf_norm_local(1) = max(linf_norm_local(1), mms_solution_es_pot)

            ! Parallel electromagnetic vector potential
            l2_err_local(2)    = l2_err_local(2) &
                               + weight &
                                * (mms_solution_a_par - A_par_in(i, k))**2
            l2_norm_local(2)   = l2_norm_local(2) &
                               + weight &
                                * mms_solution_a_par**2
            linf_err_local(2)  = max(linf_err_local(2), &
                                     abs(mms_solution_a_par - A_par_in(i, k)))
            linf_norm_local(2) = max(linf_norm_local(2), mms_solution_a_par)

            ! Parallel magnetic fluctuations (not implemented yet for vspec)
            l2_err_local(3)    = 0.0_GP
            l2_norm_local(3)   = 0.0_GP
            linf_err_local(3)  = 0.0_GP
            linf_norm_local(3) = 0.0_GP

            ! Parallel electric field
            l2_err_local(4)    = l2_err_local(4) &
                               + weight &
                                * (mms_solution_e_par - E_par_in(i, k))**2
            l2_norm_local(4)   = l2_norm_local(4) &
                               + weight &
                                * mms_solution_e_par**2
            linf_err_local(4)  = max(linf_err_local(4), &
                                     abs(mms_solution_e_par - E_par_in(i, k)))
            linf_norm_local(4) = max(linf_norm_local(4), mms_solution_e_par)
        endif
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel

        ! Communicate L2 and Linf error
        block
            ! TODO: ask how to handle the numbers here. (3->4?)
            real(kind=GP), dimension(4) :: l2_err_recv, l2_norm_recv, &
                                           linf_err_recv, linf_norm_recv
            ! Receive buffers for L2 and Linf error and norm

            call MPI_Reduce(l2_err_local, l2_err_recv, 4, MPI_GP, &
                            MPI_SUM, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(l2_norm_local, l2_norm_recv, 4, MPI_GP, &
                            MPI_SUM, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(linf_err_local, linf_err_recv, 4, MPI_GP, &
                            MPI_MAX, 0, this%dcomm_handler%get_comm_phi(), ierr)
            call MPI_Reduce(linf_norm_local, linf_norm_recv, 4, MPI_GP, &
                            MPI_MAX, 0, this%dcomm_handler%get_comm_phi(), ierr)

            if(this%dcomm_handler%is_master()) then
                do j = 3, 6
                    ! Multiply common RZ and phi weights and
                    ! take the square root for L2 norm
                    l2_err(j)  = sqrt(l2_err_recv(j - 2))
                    l2_norm(j) = sqrt(l2_norm_recv(j - 2))
                    l2_err(j)  = l2_err(j) / (l2_norm(j) + GP_EPS)

                    linf_err(j)  = linf_err_recv(j - 2)
                    linf_norm(j) = linf_norm_recv(j - 2)
                    linf_err(j)  = linf_err(j) / (linf_norm(j) + GP_EPS)
                enddo
            endif
        end block
    end subroutine
end submodule
