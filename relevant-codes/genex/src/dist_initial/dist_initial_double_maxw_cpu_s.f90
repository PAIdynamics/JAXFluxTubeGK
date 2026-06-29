submodule (dist_initial_m) dist_initial_double_maxw_cpu_s

    use math_m, only: PI
    use params_dist_initial_double_maxw_m, only: &
                                       get_params_dist_initial_double_maxw
    use params_species_m, only: get_temp_scaling
    implicit none

contains

    module subroutine initialize_double_maxw_cpu(this, &
                                                 profile_container, spec_ind)
        class(dist_initial_double_maxw_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: &
                                                        profile_container
        integer, intent(in) :: spec_ind

        call this%initialize_isotropic(profile_container, spec_ind)

        this%params = get_params_dist_initial_double_maxw(spec_ind)

    end subroutine

    real(kind = GP) module function eval_double_maxw_cpu(this, rho, &
                                        theta, phi, vp, muB) result(f_out)
        class(dist_initial_double_maxw_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi, vp, muB

        real(kind = GP) :: norm, temp, dens, vrel1, vrel2, amp1, amp2
        integer :: ierr, n

        n = this%spec_ind

        ! Evaluate density and temperature for rho, theta, phi
        call this%profile_container(n)%eval_dens(rho, theta, phi, dens, ierr)
        call this%profile_container(n)%eval_temp(rho, theta, phi, temp, ierr)
        temp = temp / get_temp_scaling(n)

        ! Apply drift in vp
        vrel1 = vp - this%params%drift_par1
        vrel2 = vp - this%params%drift_par2

        ! Get amplitudes from parameters
        amp1 = this%params%amp1
        amp2 = this%params%amp2

        ! Calculate norm
        norm = dens * (PI * temp)**(-1.5_GP) / (amp1 + amp2)

        ! Return double-Maxwellian
        f_out = norm * (amp1 * exp(-(vrel1**2 + muB) / temp) + &
                        amp2 * exp(-(vrel2**2 + muB) / temp))

    end function

end submodule
