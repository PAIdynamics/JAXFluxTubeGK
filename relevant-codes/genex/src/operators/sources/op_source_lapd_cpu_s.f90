submodule (op_source_m) op_source_lapd_cpu_s
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use params_source_lapd_m, only: get_extent, get_width, get_strength, &
                                    get_c_Te, get_prefac_Te, get_prefac_Ti
    use params_species_m, only: get_is_electrons, get_temp_scaling
    ! From PARALLAX
    use descriptors_m, only: DISTRICT_CLOSED, DISTRICT_SOL
    implicit none

contains

    elemental function lapd_profile_function(rho, c, rho_max) result(res)
        real(kind=GP), intent(in) :: rho, c, rho_max
        real(kind=GP) :: res

        if(rho < rho_max) then
            res = (1.0_GP - c) * (1.0_GP - (rho/rho_max)**2)**3 + c
        else
            res = c
        endif
    end function

    module subroutine initialize_lapd_cpu(this, mesh)
        class(op_source_lapd_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        integer :: i, k, n_points_RZ, n_points_phi
        integer, dimension(:,:), allocatable :: district
        real(kind=GP), dimension(:,:), pointer :: rho

        this%mesh => mesh
        rho => mesh%get_rho_pointer()
        this%extent   = get_extent()
        this%width    = get_width()
        this%strength = get_strength()

        n_points_RZ  = mesh%size_RZ()
        n_points_phi = mesh%size_phi()
        allocate(district(n_points_RZ, n_points_phi))

        ! Initalize district single threaded due to problem in equilibrium
        do k = 1, n_points_phi
        do i = 1, n_points_RZ
            district(i, k) = mesh%district(i, k)
        enddo
        enddo

        allocate(this%rho_buffer(n_points_RZ, n_points_phi))
        !$omp parallel default(none) &
        !$omp firstprivate(n_points_RZ, n_points_phi) &
        !$omp private(i, k) shared(this, district, rho)
        do k = 1, n_points_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            if(district(i, k) == DISTRICT_CLOSED .or. &
               district(i, k) == DISTRICT_SOL) then
                this%rho_buffer(i, k) = rho(i, k)
            else
                this%rho_buffer(i, k) = 1e100_GP
            endif
        enddo
        !$omp end do
        enddo
        !$omp end parallel
        deallocate(district)
    end subroutine

    module subroutine apply_lapd_cpu(this, da_f_out)
        class(op_source_lapd_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: i, k, l, m, n
        integer :: lb_stripped(5), ub_stripped(5)
        real(kind=GP) :: vabs2, rho, c_Te, prefac_Te, prefac_Ti, rho_max, temp
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, is_compute
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_out
        real(kind=GP), parameter :: inv_two_pi = 1.0_GP / (2.0_GP * PI)

        lb_stripped = da_f_out%get_lbound_stripped()
        ub_stripped = da_f_out%get_ubound_stripped()

        f_out => da_f_out%get_readonly_pointer()

        this%n_iterations = da_f_out%get_size_stripped()
        call this%perf_counter%start_measurement()

        absB       => this%mesh%get_absB_pointer()
        vp         => this%mesh%get_vp_pointer()
        mu         => this%mesh%get_mu_pointer()
        is_compute => this%mesh%get_is_compute_pointer()

        c_Te      = get_c_Te()
        prefac_Te = get_prefac_Te()
        prefac_Ti = get_prefac_Ti()
        rho_max   = this%mesh%rho_max()

        ! Update the f_out data in CPU memory with the values in GPU memory
        call da_f_out%update_host()

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, &
        !$omp              c_Te, prefac_Te, prefac_Ti, rho_max) &
        !$omp shared(f_out, is_compute, absB, vp, mu, this) &
        !$omp private(n, m, l, k, i, vabs2, temp)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            if(get_is_electrons(n) .eqv. .true.) then
                ! Electrons are sourced with a non-uniform temperature profile
                temp = prefac_Te * lapd_profile_function( &
                    this%rho_buffer(i, k), c_Te, rho_max)
            else
                ! Ions are sourced with a uniform temperature profile
                temp = prefac_Ti
            endif
            vabs2 = mu(m) * absB(i, k) + vp(l)**2
            temp = temp / get_temp_scaling(n)

            f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                + this%strength * inv_two_pi &
                * (PI * temp)**(-1.5_GP) * exp(-vabs2 / temp) &
                * (0.01_GP + 0.99_GP &
                           * (0.5_GP - 0.5_GP &
                                     * tanh(( this%rho_buffer(i, k) &
                                            - this%extent) &
                                            / this%width &
                                           ) &
                              ) &
                  ) &
                * is_compute(i, k)
        enddo
        !$omp end do nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        ! Update the f_out data in GPU memory with the values in CPU memory
        call da_f_out%update_device()
    end subroutine

end submodule
