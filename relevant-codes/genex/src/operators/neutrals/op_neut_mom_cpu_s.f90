submodule(op_neut_mom_m) op_neut_mom_cpu_s
    !! Contains implementation of op_neut_mom_cpu_t

    use MPI
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use params_species_m, only: get_mass, get_charge
    use params_neutrals_config_m, only: get_neut_dens_floor, get_neut_temp_floor
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use, intrinsic :: iso_fortran_env, only: INT64
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce

    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh)
        !! This subroutine initializes everything needed for the calculation of
        !! the moments of the distribution function

        class(op_neut_mom_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: n, m

        this%n_flops = 17
        this%n_ls    = 6

        this%n_sp = mesh%size_sp()
        this%mesh => mesh
        this%dcomm_handler => dcomm_handler
        call this%op_set_uniform%initialize()

        ! Calculate prefactors for moment calculation
        allocate(this%prefac_bps(this%n_sp))
        do n = 1, this%n_sp
            this%prefac_bps(n) = sqrt(2.0_GP * get_mass(n)) &
                               * get_rho_ref() / (get_charge(n) * get_L_ref())
        enddo
    end subroutine

    module subroutine apply_cpu(this, da_f_in, da_moments)
        !! This subroutine calculates the 0th, 1st and 2nd moment of the
        !! distribution function (n, Vpar, T) and saves it in moments via the
        !! 3rd index

        class(op_neut_mom_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu, vpw, muw
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, curl_normb_y
        real(kind=GP), dimension(:,:,:,:), allocatable :: send_buffer
        real(kind=GP) :: bps, area, dens_floor, temp_floor

        integer :: n_sp, lb(5), ub(5), lb_stripped(5), ub_stripped(5)
        integer :: i, k, l, m, n
        integer :: nk_stripped, k_periodic
        integer :: ierr

        n_sp = this%n_sp
        lb = da_f_in%get_lbound()
        ub = da_f_in%get_ubound()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()
        nk_stripped = ub_stripped(2) - lb_stripped(2) + 1

        f_in => da_f_in%get_readonly_pointer()
        moments => da_moments%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        ! Get pointer to mesh variables from mesh
        vp           => this%mesh%get_vp_pointer()
        mu           => this%mesh%get_mu_pointer()
        vpw          => this%mesh%get_vpw_pointer()
        muw          => this%mesh%get_muw_pointer()
        absB         => this%mesh%get_absB_pointer()
        curl_normb_y => this%mesh%get_curl_normb_y_pointer()

        ! Send buffer and moments buffer need dimensions for all species
        ! to work correctly upon sp parallelization. This is due to the
        ! collision operator needing all species in each MPI task.

        ! Allocate send_buffer
        ! Indices: 1) unstructured grid, 2) angle phi, 3) moments,
        !          4) species
        ! Moments: n, Vpar, T => 3 quantities needed.
        allocate(send_buffer(lb(1):ub(1), lb(2):ub(2), 3, this%n_sp))

        ! First touch initialize and reset buffer
        call this%op_set_uniform%apply(send_buffer, 0.0_GP)

        call profiler_start("reduction", ierr)

        ! Calculate moments. 1st loop: basic calculation
        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, nk_stripped) &
        !$omp shared(f_in, this, absB, vp, mu, vpw, muw, &
        !$omp        curl_normb_y, send_buffer) &
        !$omp private(i, k, l, m, n, k_periodic, bps, area)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb(2), ub(2)
            k_periodic = modulo(k - 1, nk_stripped) + 1
        !$omp do simd schedule(static)
        do i = lb(1), ub(1)

            ! B parallel star
            bps  = absB(i, k_periodic) + this%prefac_bps(n) * vp(l) &
                 * curl_normb_y(i, k_periodic)
            ! Vspace area element
            area = vpw(l) * muw(m) * bps * PI

            ! Density
            send_buffer(i, k, 1, n) = send_buffer(i, k, 1, n) &
                                    + area * f_in(i, k, l, m, n)
            ! Parallel drift velocity
            send_buffer(i, k, 2, n) = send_buffer(i, k, 2, n) &
                                    + area * vp(l) * f_in(i, k, l, m, n)
            ! Temperature
            send_buffer(i, k, 3, n) = send_buffer(i, k, 3, n) &
                                    + area * (vp(l)**2 + absB(i, k_periodic) &
                                              * mu(m)) * f_in(i, k, l, m, n)

        enddo
        !$omp end do simd nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("reduction", ierr)

        call profiler_start_allreduce(this%dcomm_handler%get_comm_vp_mu_sp(), &
                                      ierr)

        ! Collect results from all MPI processes
        call MPI_Allreduce(send_buffer, moments, &
                           size(send_buffer), MPI_GP, MPI_SUM, &
                           this%dcomm_handler%get_comm_vp_mu_sp(), ierr)

        call profiler_stop_allreduce(ierr)

        dens_floor = get_neut_dens_floor()
        temp_floor = get_neut_temp_floor()

        call profiler_start("store", ierr)

        ! 2nd loop: apply corrections
        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped, n_sp) &
        !$omp shared(this, moments, dens_floor, temp_floor) &
        !$omp private(n, k, i)
        do n = 1, n_sp
        do k = lb(2), ub(2)
        !$omp do simd schedule(static)
        do i = lb(1), ub(1)

            ! Density floor: take larger value (i.e. prevent value zero)
            moments(i, k, 1, n) = max(moments(i, k, 1, n), dens_floor)
            ! V par normalized by density
            moments(i, k, 2, n) = moments(i, k, 2, n) / moments(i, k, 1, n)
            ! 2nd moment normalized by density + correction by
            ! V par squared
            moments(i, k, 3, n) = max(2.0_GP / 3.0_GP * &
                                        (moments(i, k, 3, n) / &
                                         moments(i, k, 1, n) - &
                                         moments(i, k, 2, n)**2), &
                                      temp_floor)
        enddo
        !$omp end do simd nowait
        enddo
        enddo
        !$omp end parallel

        call profiler_stop("store", ierr)

        deallocate(send_buffer)
        call this%perf_counter%end_measurement()

    end subroutine

end submodule
