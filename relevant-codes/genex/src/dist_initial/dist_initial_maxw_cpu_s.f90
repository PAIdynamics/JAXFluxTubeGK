submodule (dist_initial_m) dist_initial_maxw_cpu_s

    use math_m, only: PI
    use params_species_m, only: get_temp_scaling

    implicit none

contains

    module subroutine initialize_maxw_cpu(this, profile_container, &
                                                spec_ind)
        class(dist_initial_maxw_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: &
                                                        profile_container
        integer, intent(in) :: spec_ind

        call this%initialize_isotropic(profile_container, spec_ind)

    end subroutine

    real(kind = GP) module function eval_maxw_cpu(this, rho, theta, &
                                        phi, vp, muB) result(f_out)
        class(dist_initial_maxw_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi, vp, muB

        real(kind = GP) :: norm, dens, temp
        integer :: ierr, n

        n = this%spec_ind

        ! Evaluate density and temperature for rho, theta, phi
        call this%profile_container(n)%eval_dens(rho, theta, phi, dens, ierr)
        call this%profile_container(n)%eval_temp(rho, theta, phi, temp, ierr)
        temp = temp / get_temp_scaling(n)

        ! Return Maxwellian in 3D cartesian coordinate system in 2D velspace
        ! for given density and temperature
        norm = dens * (PI * temp)**(-1.5_GP)
        f_out = norm * exp(-(muB + vp**2) / temp)

    end function

end submodule
