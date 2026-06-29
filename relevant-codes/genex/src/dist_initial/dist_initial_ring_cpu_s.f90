submodule (dist_initial_m) dist_initial_ring_cpu_s

    use math_m, only: PI
    use params_dist_initial_ring_m, only: get_params_dist_initial_ring
    implicit none

contains

    module subroutine initialize_ring_cpu(this, profile_container, spec_ind)
        class(dist_initial_ring_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: &
                                                        profile_container
        integer, intent(in) :: spec_ind

        call this%initialize_isotropic(profile_container, spec_ind)

        this%params = get_params_dist_initial_ring(spec_ind)

    end subroutine

    real(kind = GP) module function eval_ring_cpu(this, rho, theta, phi, vp, &
                                                  muB) result(f_out)
        class(dist_initial_ring_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi, vp, muB

        real(kind = GP) :: norm, dens, vrel, muBrel, dperp, wpar, wperp
        integer :: ierr, n

        n = this%spec_ind

        ! Evaluate density for rho, theta, phi
        call this%profile_container(n)%eval_dens(rho, theta, phi, dens, ierr)

        ! Get parameter
        dperp = this%params%drift_perp
        wpar  = this%params%width_par
        wperp = this%params%width_perp

        ! Apply drift in vp
        vrel = vp - this%params%drift_par
        ! Apply drift in muB
        muBrel = (sqrt(muB) - dperp)**2

        ! Return ring in 3D cartesian coordinate system in
        ! 2D velspace for given density
        norm = dens / (PI**2 * wpar * wperp * sqrt(muB) * &
                        (1.0_GP + erf(dperp / wperp)))
        f_out = norm * exp(-(vrel / wpar)**2 - muBrel / wperp**2)

    end function

end submodule
