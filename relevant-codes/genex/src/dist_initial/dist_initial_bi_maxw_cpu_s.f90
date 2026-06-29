submodule (dist_initial_m) dist_initial_bi_maxw_cpu_s

    use math_m, only: PI
    use params_dist_initial_bi_maxw_m, only: &
                                  get_params_dist_initial_bi_maxw
    use params_species_m, only: get_temp_scaling
    implicit none

contains

    module subroutine initialize_bi_maxw_cpu(this, profile_container, &
                                             spec_ind)
        class(dist_initial_bi_maxw_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: profile_container
        integer, intent(in) :: spec_ind

        call this%initialize_anisotropic(profile_container, spec_ind)

        this%params = get_params_dist_initial_bi_maxw(spec_ind)

    end subroutine

    real(kind = GP) module function eval_bi_maxw_cpu(this, rho, &
                                        theta, phi, vp, muB) result(f_out)
        class(dist_initial_bi_maxw_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi, vp, muB

        real(kind = GP) :: norm, vrel, dens, temp_par, temp_perp
        integer :: ierr, n

        n = this%spec_ind

        associate(pc => this%profile_container)
            ! Evaluate density for rho, theta, phi
            call pc(n)%eval_dens(rho, theta, phi, dens, ierr)

            ! Evaluate temperature for rho, theta, phi
            if(pc(n)%get_is_isotropic()) then
                ! Isotropic, par and perp are the same
                call pc(n)%eval_temp(rho, theta, phi, temp_par, ierr)
                call pc(n)%eval_temp(rho, theta, phi, temp_perp, ierr)
            else
                ! Anisotropic, par and perp are different
                call pc(n)%eval_temp_par(rho, theta, phi, temp_par, ierr)
                call pc(n)%eval_temp_perp(rho, theta, phi, temp_perp, ierr)
            end if
        end associate

        ! Parallel velocity relative to drift velocity specified by param
        vrel = vp - this%params%drift_vpar
        ! Flat factors on parallel and perpendicular temperatures
        temp_par  = temp_par / get_temp_scaling(n) * this%params%tau_par
        temp_perp = temp_perp / get_temp_scaling(n) * this%params%tau_perp

        ! Return bi-Maxwellian in 3D cartesian coordinate system in
        ! 2D velspace for given density and anisotropic temperature
        norm = dens / (PI**(1.5_GP) * sqrt(temp_par) * temp_perp)
        f_out = norm * exp(- vrel**2 / temp_par - muB / temp_perp)

    end function

end submodule
